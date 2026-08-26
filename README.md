![image](https://github.com/user-attachments/assets/97fe2564-a749-4ea9-8221-30e52db32d46)
![image](https://github.com/user-attachments/assets/73b2cd44-6940-4e9e-beff-a02e03612d46)

# ArchaicSeeker3.1-mamba

This project is developed and maintained by the **[Shuhua Xu's Research Group](https://pog.fudan.edu.cn/)**, School of Life Sciences, Fudan University.

## About

`ArchaicSeeker3.1-mamba` is an algorithm for detecting archaic introgression segments (e.g., from Neanderthals and Denisovans) in modern human genomes. It is based on the Mamba (SSM-Mamba) architecture, designed for accurate and efficient analysis of large-scale genomic data.

This repository provides the core software and example scripts to demonstrate how to use it for parallel analysis on multi-GPU systems.

---

## Citation

If you use `ArchaicSeeker3.1-mamba` in your research, please cite our relevant publications. For a list of publications, please visit our group's website: [POG Fudan Publications](https://pog.fudan.edu.cn/#/article).

> **Citation:**
> unpublished

---

## License

This project is licensed under the **ArchaicSeeker Academic Use License**.
-   **For Academic Users:** Free to use, modify, and distribute for non-commercial research purposes.
-   **For Commercial Users:** A separate commercial license is required.

Please see the `LICENSE` file for detailed terms.

---

## Installation

The automated installation script is recommended for most users.

### Prerequisites

* A Linux-based operating system.
* **Conda** or **Miniforge/Mamba** installed.
* For GPU acceleration: An **NVIDIA GPU** with the appropriate **CUDA Toolkit** and drivers installed.

---

### Recommended Installation via Script

This method uses the provided `install.sh` script to automatically create a conda environment and handle all dependencies, including complex ones.

1.  **Run the installation script:**
    ```bash
    chmod +x install.sh
    ./install.sh
    ```
    The script will create a new conda environment named `as3_mamba`.

2.  **Activate the environment:**
    ```bash
    conda activate as3_mamba
    ```

> **📦 Offline or Difficult Installations:**
> The `mamba-ssm` and `causal-conv1d` packages can be difficult to build from source. For convenience, you can **pre-download their `.whl` files** that match your system (Python 3.9, CUDA version) and place them in the root directory of this project. The `install.sh` script will automatically detect and install them.

---

## Data Preparation

For convenience, we provide ready-to-use reference data. With the preprocessed
Ref1028 panel, users only need to prepare a phased input VCF for the target
samples. A custom reference panel is needed only when the supplied panel is not
suitable for the intended analysis.

* Download the data for either **GRCh38** or **CHM13** from Zenodo record:
    **[https://zenodo.org/records/14552025](https://zenodo.org/records/14552025)**

### Direct downloads from the AS3 web service

The following GRCh38 data are available as public, read-only static downloads and can be used directly without an AS3 web account:

#### Ref1028 reference panel

**[Browse the Ref1028 release](https://pog.fudan.edu.cn/as3-downloads/reference-panel/Ref1028/)**

This is the exact chromosome-wise panel currently configured in the AS3 web service. For chromosome `N`, download all of the following:

```text
Ref_Panel.chrN.vcf.gz
Ref_Panel.chrN.vcf.gz.tbi
Ref_Panel.map.txt
```

Example for chromosome 22:

```bash
wget https://pog.fudan.edu.cn/as3-downloads/reference-panel/Ref1028/Ref_Panel.chr22.vcf.gz
wget https://pog.fudan.edu.cn/as3-downloads/reference-panel/Ref1028/Ref_Panel.chr22.vcf.gz.tbi
wget https://pog.fudan.edu.cn/as3-downloads/reference-panel/Ref1028/Ref_Panel.map.txt
```

The VCF can be supplied directly to `--reference`, and `Ref_Panel.map.txt` can be supplied directly to `--map`.

The panel contains four high-coverage archaic genomes—`AltaiNeandertal`,
`Vindija33.19`, `Chagyrskaya-Phalanx`, and `Denisova`—together with the modern
human reference samples used by AS3.

After downloading the chromosome-specific panel and map into the current
directory, the only analysis input that users need to supply is their phased
target VCF. For example, for chromosome 22:

```bash
TARGET_VCF=/path/to/input.phased.chr22.vcf.gz
CUDA_VISIBLE_DEVICES=0 python ArchaicSeeker3.1-mamba \
  -t "${TARGET_VCF}" \
  -r Ref_Panel.chr22.vcf.gz \
  -m Ref_Panel.map.txt \
  -o as3_chr22
```

#### GRCh38 3N1D high-quality-region masks

**[Browse the GRCh38 3N1D mask release](https://pog.fudan.edu.cn/as3-downloads/high-quality-mask/GRCh38/3N1D/)**

The source FilterBeds for three high-coverage Neanderthal genomes (Altai, Vindija33.19 and Chagyrskaya) and one Denisovan genome were lifted from hg19 to GRCh38. For each chromosome, the four lifted BEDs were concatenated, coordinate-sorted and combined with `bedtools merge`; these files therefore represent the merged union of the four source pass-interval sets.

Two equivalent chromosome naming styles are provided:

```text
chrN_mask.bed       # chromosome column: 1, 2, ...
chrN_mask.wchr.bed  # chromosome column: chr1, chr2, ...
```

Choose the file matching the chromosome convention in your VCF. The mask is directly usable by interval-aware tools, for example:

```bash
bcftools view -T chr22_mask.wchr.bed -Oz -o input.high_quality.vcf.gz input.vcf.gz
```

The bundled `00.preprocess.sh` example currently expects separate `N_chrN.bed` and `D_chrN.bed` files. To use the combined 3N1D release, apply the BED directly as above or adapt that mask stage to use one combined file.

These masks are intended for users building or filtering a custom reference
panel. They are not required when using the downloadable preprocessed Ref1028
panel.

Each release directory contains a `README.txt` and `SHA256SUMS`. Static downloads support HTTP range requests, so interrupted downloads can be resumed.

### AS3 Web for single-chromosome analyses

For a single chromosome, we recommend the hosted
**[ArchaicSeeker3 Web service](https://pog.fudan.edu.cn/as3web/)**. Register an
account, wait for administrator approval, upload a phased chromosome VCF, and
submit a `Real AS3` task with the supplied reference panel. The web service is
the simplest option when local GPU setup or custom reference preparation is not
required.

---

## Usage Workflow

The recommended path is to use the downloadable Ref1028 VCF and map directly,
as in the chromosome 22 quick-start command above. No reference-panel
preprocessing is required.

Use `00.preprocess.sh` only when building a custom reference panel. Configure
all paths in its **User Configuration** section before running it; the repository
does not bundle the placeholder `examples/raw_data` files. The script produces
`Final_Target_VCFs`, `Final_Ref_VCFs`, and `reference.map`.

For a configured multi-chromosome dataset, `01.run_archaicseeker3.sh` provides
parallel GPU execution. Set its input/output paths and GPU list, then run:

```bash
bash 01.run_archaicseeker3.sh
```

### Direct Execution (Advanced)

For debugging or running on a single file, you can execute the main program directly.

**Basic Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python ArchaicSeeker3.1-mamba \
    -t <path/to/target.vcf.gz> \
    -r <path/to/reference.vcf.gz> \
    -m <path/to/map.txt> \
    -o <path/to/output_folder> \
    [OPTIONS]
```

**Command-Line Arguments:**

| Argument | Shorthand | Description | Default |
| :--- | :--- | :--- | :--- |
| **`--test-mixed`** | `-t` | **Required.** Path to the phased VCF file of the target samples. | `None` |
| **`--reference`** | `-r` | **Required.** Path to the reference panel VCF file (archaic & African). | `None` |
| **`--map`** | `-m` | **Required.** Path to the reference map file. | `None` |
| **`--out-folder`** | `-o` | **Required.** Path to the folder where results will be saved. | `None` |
| `--base-model-cp`| | Path to the base model checkpoint (`.pth`). | Defaults to `./exp/Basemodel.../best_model.pth` |
| `--smoother-model-cp`| | Path to the smoother model checkpoint (`.pth`). | Defaults to `./exp/Smoother.../best_model.pth` |
| `--stride` | | The stride of the sliding window for model inference. | `512` |
| `--merge` | | Maximum gap (bp) for the canonical exact merge. | `10000` |
| `--anc` | | Archaic parameter setting for analysis. | `0` |
| `--target-chunk-size`| | Process target samples in chunks of this size to reduce memory usage. `None` means all at once. | `None` |
| `--base-model-args`| | Path to the base model's arguments file (`.pckl`). If `None`, auto-detected. | `None` |
| `--smoother-model-args`| | Path to the smoother model's arguments file (`.pckl`). If `None`, auto-detected. | `None` |

## Output Format

AS3 now performs its standard BED construction internally. The default command:

1. retains raw segments with length `>= 5 kb` and score `>= 0`;
2. exact-merges retained segments with the `--merge` distance (default `10 kb`);
3. recalculates ancestry label, SNP counts, and mean score from the raw SNP details;
4. writes the combined result and ancestry-specific BED files.

The standard output files are:

```text
introgression.bed
introgression.denisovan.bed
introgression.neanderthal.bed
introgression.mosaic.bed
introgression.raw.bed
introgression.raw.snps.gz
introgression_prediction.txt
run.log
```

### 1. `introgression.bed`

This is the canonical combined result after the standard 5 kb pre-filter and
exact merge. The three ancestry-specific BEDs use the same format and contain
only the corresponding rows.

| Column | Description |
| :--- | :--- |
| **Chr** | Chromosome |
| **Start** | First supporting VCF `POS` value (1-based, inclusive) |
| **End** | Last supporting VCF `POS` value (1-based, inclusive) |
| **Haplotype** | Internal numeric haplotype index used during inference. |
| **Archaic** | Predicted source: `1`=Denisovan, `2`=Neanderthal, `3`=Mosaic |
| **#SNP** | Number of SNPs within the segment |
| **Score** | Mean score recalculated from the SNPs supporting the final merged label. A higher score indicates higher confidence. |
| **#SNP_Archaic1** | Number of SNPs supporting Archaic Source 1 |
| **#SNP_Archaic2** | Number of SNPs supporting Archaic Source 2 |
| **SampleID_HapID** | A globally unique identifier combining the original Sample ID and haplotype (1 or 2). |

Although these files use a `.bed` suffix, AS3 preserves the 1-based VCF SNP
positions rather than converting them to standard 0-based, half-open BED
coordinates. Historical AS3 length thresholds are therefore evaluated exactly
as `End - Start`, as shown below.

### 2. Raw and SNP-level outputs

`introgression.raw.bed` preserves the unmerged segment calls, and
`introgression.raw.snps.gz` preserves the SNP-level evidence used for exact
merge and custom reprocessing. These files are not overwritten by the standard
post-processing step.

`introgression_prediction.txt` provides the SNP-level ancestry matrix:

-   **Rows**: Haplotype indices.
-   **Columns**: Variant positions.
-   **Values**: Predicted ancestry: `0`=African (non-introgressed), `1`=Denisovan, `2`=Neanderthal.

### 3. Optional strict filtering

The historical high-confidence tract definition (`length >= 15 kb` and
`score >= 0.85`) is intentionally not part of the default AS3 algorithm. It is
a direct row filter on the already merged `introgression.bed`; it does not merge
segments or recalculate scores:

```bash
awk 'BEGIN{OFS="\t"} ($3-$2)>=15000 && $7>=0.85' \
  introgression.bed > introgression.strict.bed
```

## Contact

For questions, bug reports, or collaboration inquiries, please visit our lab website:
**[https://pog.fudan.edu.cn/](https://pog.fudan.edu.cn/)**
