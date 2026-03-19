#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
from pathlib import Path
import subprocess
import pandas as pd

ACC_PATH = "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/acc.py"
ACC_DIR = os.path.dirname(ACC_PATH)
sys.path.append(ACC_DIR)
from acc import cal_accuracy_hap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Task for ArchaicSeeker3")
    parser.add_argument("--seed", type=str, required=True, help="seed")
    parser.add_argument("--min_length", type=float, required=True, help="min_length_in_kb")
    parser.add_argument("--min_score", type=float, required=True, help="min_score")
    parser.add_argument("--distance", type=int, required=True, help="distance")
    return parser.parse_args()

# def Filter(AS3_raw_bed,AS3_filtered_bed,min_length,min_score):
#     df = pd.read_csv(AS3_raw_bed, sep="\t", header=None)
#     df = df[df[8] >= min_score]
#     df = df[df[2] - df[1] >= min_length * 1000]
#     df.to_csv(AS3_filtered_bed, sep="\t", header=None, index=False)
#     return df

# def Archaic2DenNean(Archaic_infered_bed ,Archaic_infered_tmp_bed,Den_infered_bed, Nean_infered_bed):
#     df = pd.read_csv(Archaic_infered_bed, sep="\t", header=None)
#     df[df[6] == 1][[0,1,2,5,6,7,8]].to_csv(Den_infered_bed, sep="\t", header=None, index=False)
#     df[df[6] == 2][[0,1,2,5,6,7,8]].to_csv(Nean_infered_bed, sep="\t", header=None, index=False)
#     df[[0,1,2,5,6,7,8]].to_csv(Archaic_infered_tmp_bed,sep='\t',header=None,index=False)

def Filter(AS3_raw_bed,AS3_filtered_bed,min_length,min_score):
    df = pd.read_csv(AS3_raw_bed, sep="\t", header=None)
    df = df[df[6] >= min_score]
    df = df[df[2] - df[1] >= min_length * 1000]
    df.to_csv(AS3_filtered_bed, sep="\t", header=None, index=False)
    return df

def Archaic2DenNean(Archaic_infered_bed ,Archaic_infered_tmp_bed,Den_infered_bed, Nean_infered_bed):
    df = pd.read_csv(Archaic_infered_bed, sep="\t", header=None)
    df[df[4] == 1].to_csv(Den_infered_bed, sep="\t", header=None, index=False)
    df[df[4] == 2].to_csv(Nean_infered_bed, sep="\t", header=None, index=False)
    df.to_csv(Archaic_infered_tmp_bed,sep='\t',header=None,index=False)

def main():
    args = parse_args()
    seed = args.seed
    min_length = args.min_length
    min_score = args.min_score
    distance = args.distance

    basedir = Path("/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut/")
    out_dir = basedir / str(seed)
    prefix = f"temp_{min_length}kb_s{min_score}_d{distance}"

    print(f"seed: {seed}")
    print(f"prefix: {prefix}")

    Archaic_sim_bed = out_dir / "Archaic.sim.bed"
    Den_sim_bed     = out_dir / "Den.sim.bed"
    Nean_sim_bed    = out_dir / "Nean.sim.bed"

    AS3_raw_bed = out_dir / "introgression_prediction.raw.bk.bed"
    # AS3_raw_snp = out_dir / "introgression_prediction.raw.snp.bk.gz"
    # AS3_raw_prob = out_dir / "introgression_prob_matrix.bk.txt"

    AS3_filtered_bed = out_dir / f"introgression_prediction.raw.bed"
    # AS3_filtered_snp = out_dir / f"introgression_prediction.raw.snp.gz"
    # AS3_filtered_prob = out_dir / f"introgression_prob_matrix.txt"

    Filter(AS3_raw_bed, AS3_filtered_bed, min_length, min_score)

    Archaic_infered_bed = out_dir / f"{prefix}.bed"
    Archaic_infered_tmp_bed = out_dir / f"Archaic.{prefix}.bed"
    Den_infered_bed     = out_dir / f"Den.{prefix}.bed"
    Nean_infered_bed    = out_dir / f"Nean.{prefix}.bed"

    out_accuracy = out_dir / f"{prefix}.accuracy"

    # merge segments
    cmd = (
        f"cd {out_dir} && "
        f"/home/linhuanyu/02_Software/ArchaicSeeker3_memFast_TwoStage/merge_bed_segments.py "
        f"-i {AS3_filtered_bed} "
        f"-o {Archaic_infered_bed} -d {distance}"
    )
    subprocess.run(cmd, shell=True, check=True)

    # split Den / Nean
    Archaic2DenNean(Archaic_infered_bed, Archaic_infered_tmp_bed, Den_infered_bed, Nean_infered_bed)

    # accuracy
    Archaic_prec, Archaic_rec, Archaic_f1, Archaic_ratio = cal_accuracy_hap(Archaic_sim_bed, Archaic_infered_tmp_bed)
    Den_prec, Den_rec, Den_f1, Den_ratio                 = cal_accuracy_hap(Den_sim_bed,     Den_infered_bed)
    Nean_prec, Nean_rec, Nean_f1, Nean_ratio             = cal_accuracy_hap(Nean_sim_bed,    Nean_infered_bed)

    with open(out_accuracy, "w") as f:
        f.write(
            f"{prefix}\t{seed}\t"
            f"{Archaic_prec}\t{Archaic_rec}\t{Archaic_f1}\t{Archaic_ratio}\t"
            f"{Den_prec}\t{Den_rec}\t{Den_f1}\t{Den_ratio}\t"
            f"{Nean_prec}\t{Nean_rec}\t{Nean_f1}\t{Nean_ratio}\n"
        )


if __name__ == "__main__":
    main()
