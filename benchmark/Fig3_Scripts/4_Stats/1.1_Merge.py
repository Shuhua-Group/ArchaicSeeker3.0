import pandas as pd
import numpy as np
import argparse
import os
import subprocess
from pathlib import Path
import sys, os
ACC_PATH = "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/acc.py"
ACC_DIR = os.path.dirname(ACC_PATH)
sys.path.append(ACC_DIR)

from acc import cal_accuracy_hap, cal_accuracy_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Task for ArchaicSeeker3")
    parser.add_argument("--seed", type=str, required=True, help="seed")
    parser.add_argument("--prefix", type=str, required=True, help="prefix")
    return parser.parse_args()

def Archaic2DenNean(Archaic_infered_bed, Den_infered_bed, Nean_infered_bed):
    df = pd.read_csv(Archaic_infered_bed, sep="\t", header=None)
    df[df[4] == 1].to_csv(Den_infered_bed, sep="\t", header=None, index=False)
    df[df[4] == 2].to_csv(Nean_infered_bed, sep="\t", header=None, index=False)

def main():
    args = parse_args()
    seed = args.seed
    prefix = args.prefix
    
    basedir = Path("/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut/")
    out_dir = basedir / str(seed)

    print(f"seed: {seed}")
    print(f"prefix: {prefix}")

    Archaic_sim_bed = out_dir / f"Archaic.sim.bed"
    Archaic_sim_info_bed = out_dir / f"Archaic.sim.info.bed"
    Den_sim_bed = out_dir / f"Den.sim.bed"
    Nean_sim_bed = out_dir / f"Nean.sim.bed"
    Archaic_infered_bed = out_dir / f"{prefix}.bed"
    Den_infered_bed = out_dir / f"Den.{prefix}.bed"
    Nean_infered_bed = out_dir / f"Nean.{prefix}.bed"

    out_accuracy = out_dir / f"{prefix}.accuracy"
    
    Archaic2DenNean(Archaic_infered_bed, Den_infered_bed, Nean_infered_bed)

    Archaic_prec, Archaic_rec, Archaic_f1, Archaic_ratio = cal_accuracy_hap(Archaic_sim_bed, Archaic_infered_bed)  
    Den_prec, Den_rec, Den_f1, Den_ratio = cal_accuracy_hap(Den_sim_bed, Den_infered_bed)
    Nean_prec, Nean_rec, Nean_f1, Nean_ratio = cal_accuracy_hap(Nean_sim_bed, Nean_infered_bed)

    with open(out_accuracy, "w") as f:
        f.write(f"{prefix}\t{seed}\t{Archaic_prec}\t{Archaic_rec}\t{Archaic_f1}\t{Archaic_ratio}\t{Den_prec}\t{Den_rec}\t{Den_f1}\t{Den_ratio}\t{Nean_prec}\t{Nean_rec}\t{Nean_f1}\t{Nean_ratio}\n")
    

if __name__ == "__main__":
    main()




