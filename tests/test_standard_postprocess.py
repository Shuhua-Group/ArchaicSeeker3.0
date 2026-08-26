from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.standard_postprocess import (
    ANCESTRY_BED_NAMES,
    BED_COLUMNS,
    COMBINED_BED_NAME,
    RAW_BED_NAME,
    RAW_SNPS_NAME,
    run_standard_postprocess,
)


class StandardPostprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        raw_rows = [
            [22, 100, 5100, 0, 1, 2, 0.90, 2, 0, "sampleA_1"],
            [22, 6000, 12000, 0, 2, 2, 0.80, 0, 2, "sampleA_1"],
            [22, 20000, 24000, 0, 1, 2, 0.99, 2, 0, "short_1"],
            [22, 30000, 36000, 1, 2, 2, -0.10, 0, 2, "negative_1"],
            [22, 50000, 56000, 2, 2, 2, 0.70, 0, 2, "sampleB_1"],
            [22, 70000, 76000, 3, 1, 2, 0.95, 2, 0, "sampleC_1"],
        ]
        self.raw_bed_path = self.output_dir / RAW_BED_NAME
        pd.DataFrame(raw_rows).to_csv(
            self.raw_bed_path, sep="\t", header=False, index=False
        )
        self.raw_bytes = self.raw_bed_path.read_bytes()

        snp_rows = [
            self._snp_row(22, "sampleA_1", 100, 5100, "100,5100", "1,1", "0.9,0.8", "0.1,0.2"),
            self._snp_row(22, "sampleA_1", 6000, 12000, "6000,12000", "2,2", "0.2,0.3", "0.7,0.6"),
            self._snp_row(22, "short_1", 20000, 24000, "20000,24000", "1,1", "0.9,0.9", "0.1,0.1"),
            self._snp_row(22, "negative_1", 30000, 36000, "30000,36000", "2,2", "0.1,0.1", "0.9,0.9"),
            self._snp_row(22, "sampleB_1", 50000, 56000, "50000,56000", "2,2", "0.2,0.2", "0.7,0.8"),
            self._snp_row(22, "sampleC_1", 70000, 76000, "70000,76000", "1,1", "0.95,0.85", "0.05,0.15"),
        ]
        with gzip.open(self.output_dir / RAW_SNPS_NAME, "wt", encoding="utf-8") as handle:
            pd.DataFrame(snp_rows).to_csv(handle, sep="\t", index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _snp_row(chromosome, sample, start, end, positions, states, prob1, prob2):
        return {
            "chr": chromosome,
            "sample_hap_id": sample,
            "start_pos": start,
            "end_pos": end,
            "snp_positions": positions,
            "snp_states": states,
            "snp_prob_label1": prob1,
            "snp_prob_label2": prob2,
        }

    def test_default_filter_exact_merge_and_split(self) -> None:
        summary = run_standard_postprocess(self.output_dir, merge_distance=10_000)

        self.assertEqual(summary["raw_segments"], 6)
        self.assertEqual(summary["filtered_segments"], 4)
        self.assertEqual(summary["merged_segments"], 3)
        self.assertEqual(summary["denisovan_segments"], 1)
        self.assertEqual(summary["neanderthal_segments"], 1)
        self.assertEqual(summary["mosaic_segments"], 1)
        self.assertEqual(self.raw_bed_path.read_bytes(), self.raw_bytes)

        combined = pd.read_csv(
            self.output_dir / COMBINED_BED_NAME,
            sep="\t",
            header=None,
            names=BED_COLUMNS,
        )
        self.assertEqual(set(combined["ancestry_label"]), {1, 2, 3})

        mosaic = combined.loc[combined["ancestry_label"] == 3].iloc[0]
        self.assertEqual(mosaic["sample_hap_id"], "sampleA_1")
        self.assertEqual(int(mosaic["start_pos"]), 100)
        self.assertEqual(int(mosaic["end_pos"]), 12000)
        self.assertEqual(int(mosaic["num_snps"]), 4)
        self.assertAlmostEqual(float(mosaic["avg_prob"]), 0.75)

        for label, filename in ANCESTRY_BED_NAMES.items():
            split = pd.read_csv(self.output_dir / filename, sep="\t", header=None)
            self.assertEqual(len(split), 1)
            self.assertEqual(int(split.iloc[0, 4]), label)

    def test_zero_merge_distance_keeps_filtered_segments_separate(self) -> None:
        summary = run_standard_postprocess(self.output_dir, merge_distance=0)
        self.assertEqual(summary["filtered_segments"], 4)
        self.assertEqual(summary["merged_segments"], 4)
        self.assertEqual(self.raw_bed_path.read_bytes(), self.raw_bytes)

    def test_empty_raw_bed_writes_all_empty_canonical_outputs(self) -> None:
        self.raw_bed_path.write_bytes(b"")
        (self.output_dir / RAW_SNPS_NAME).unlink()

        summary = run_standard_postprocess(self.output_dir, merge_distance=10_000)

        self.assertEqual(summary["raw_segments"], 0)
        self.assertEqual(summary["merged_segments"], 0)
        self.assertEqual((self.output_dir / COMBINED_BED_NAME).read_bytes(), b"")
        for filename in ANCESTRY_BED_NAMES.values():
            self.assertEqual((self.output_dir / filename).read_bytes(), b"")

    def test_combined_output_matches_legacy_external_pipeline(self) -> None:
        run_standard_postprocess(self.output_dir, merge_distance=10_000)
        internal_bytes = (self.output_dir / COMBINED_BED_NAME).read_bytes()

        legacy_raw = self.output_dir / "legacy.raw.bed"
        legacy_snps = self.output_dir / "legacy.raw.snps.gz"
        legacy_output = self.output_dir / "legacy.bed"

        raw = pd.read_csv(self.raw_bed_path, sep="\t", header=None)
        raw = raw.loc[(raw[2] - raw[1] >= 5_000) & (raw[6] >= 0)]
        raw.to_csv(legacy_raw, sep="\t", header=False, index=False)
        shutil.copy2(self.output_dir / RAW_SNPS_NAME, legacy_snps)

        merge_script = Path(__file__).parents[1] / "src" / "merge_bed_segments.py"
        subprocess.run(
            [
                sys.executable,
                str(merge_script),
                "-i",
                str(legacy_raw),
                "-o",
                str(legacy_output),
                "-d",
                "10000",
            ],
            check=True,
        )

        self.assertEqual(internal_bytes, legacy_output.read_bytes())


if __name__ == "__main__":
    unittest.main()
