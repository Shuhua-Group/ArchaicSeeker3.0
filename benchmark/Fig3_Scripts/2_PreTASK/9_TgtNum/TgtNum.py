import pandas as pd
import numpy as np
import argparse
import os
import subprocess
from pathlib import Path

def process_Den_bed(Den1_bed_path, Den2_bed_path, Den_bed_path_out,tgtNum):
    Den1_bed_raw = pd.read_csv(Den1_bed_path, sep="\t", header=None)
    Den1_bed_raw = Den1_bed_raw[Den1_bed_raw[3] <= 2 * tgtNum - 1]
    Den2_bed_raw = pd.read_csv(Den2_bed_path, sep="\t", header=None)
    Den2_bed_raw = Den2_bed_raw[Den2_bed_raw[3] <= 2 * tgtNum - 1]
    Den_bed_raw = pd.concat([Den1_bed_raw, Den2_bed_raw])
    Den_bed_raw.to_csv(Den_bed_path_out, sep="\t", header=False, index=False)

def process_Nean_bed(Nean_bed_path, Nean_bed_path_out,tgtNum):
    Nean_bed_raw = pd.read_csv(Nean_bed_path, sep="\t", header=None)
    Nean_bed_raw = Nean_bed_raw[Nean_bed_raw[3] <= 2 * tgtNum - 1]
    Nean_bed_raw.to_csv(Nean_bed_path_out, sep="\t", header=False, index=False)

def process_Archaic_bed(Den_bed_path, Nean_bed_path, archaic_bed_path, archaic_bed_info_path):
    Den_bed_raw = pd.read_csv(Den_bed_path, sep="\t", header=None)
    Den_bed_raw[4] = "Den"
    Nean_bed_raw = pd.read_csv(Nean_bed_path, sep="\t", header=None)
    Nean_bed_raw[4] = "Nean"
    df = pd.concat([Den_bed_raw, Nean_bed_raw])

    merged_intervals = []

    # 按 chr + 倒数两列 来分组
    for (chrom, col3, col4), group in df.groupby([0, 3, 4]):
        group = group.sort_values(by=1)  # start 排序
        current_start = None
        current_end = None

        for _, row in group.iterrows():
            start, end = row[1], row[2]

            if current_start is None:
                current_start, current_end = start, end
            else:
                # 合并重叠/相邻的区间
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    merged_intervals.append([chrom, current_start, current_end, col3, col4])
                    current_start, current_end = start, end

        merged_intervals.append([chrom, current_start, current_end, col3, col4])

    merged_bed = pd.DataFrame(merged_intervals, columns=[0, 1, 2, 3, 4])

    merged_bed = merged_bed.sort_values(by=[3, 1]).reset_index(drop=True)

    merged_bed[[0,1,2,3]].to_csv(archaic_bed_path, sep="\t", header=False, index=False)

    merged_bed.columns = ["Chr", "Start", "End", "Index","Archaic"]
    merged_bed['Archaic'].replace(1, "Den", inplace=True)
    merged_bed['Archaic'].replace(2, "Nean", inplace=True)
    merged_bed['Length_kb'] = (merged_bed['End'] - merged_bed['Start']) / 1000
    merged_bed.to_csv(archaic_bed_info_path, sep="\t", header=False, index=False)
        
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Task for ArchaicSeeker3")
    parser.add_argument("--seed", type=str, required=True, help="seed")
    return parser.parse_args()

def main():
    args = parse_args()
    seed = args.seed
    print(f"seed: {seed}")

    # 输入文件夹
    indir = Path(f"/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/4_MoreSeg/{seed}")
    # 输出文件夹
    outdir = Path(f"/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/9_TgtNum/{seed}")

    # 创建输出文件夹
    os.makedirs(outdir, exist_ok=True)

    # 原始模拟数据

    Den1_bed_path = indir / "Den1.sim.bed"
    Den2_bed_path = indir / "Den2.sim.bed"
    Nean_bed_path = indir / "Nean.sim.bed"
    
    for tgtNum in [1, 10, 25, 50, 100, 250, 500,1000]:
        Den_bed_path_out = outdir / f"Den.sim.{tgtNum}.bed"
        Nean_bed_path_out = outdir / f"Nean.sim.{tgtNum}.bed"
        Archaic_bed_path = outdir / f"Archaic.sim.{tgtNum}.bed"
        Archaic_bed_info_path = outdir / f"Archaic.sim.{tgtNum}.info.bed"
        process_Den_bed(Den1_bed_path, Den2_bed_path, Den_bed_path_out,tgtNum)
        process_Nean_bed(Nean_bed_path, Nean_bed_path_out,tgtNum)
        process_Archaic_bed(Den_bed_path_out, Nean_bed_path_out, Archaic_bed_path, Archaic_bed_info_path)

    # Target_vcf

    Target_vcf = indir / "target.vcf.gz"
    Target_vcf_path = outdir / "target1000.vcf.gz"
    subprocess.run(f"cp {Target_vcf} {Target_vcf_path}", shell=True)
    subprocess.run(f"bcftools index -t {Target_vcf_path}", shell=True)
    Target_vcf_tmp = Target_vcf_path
    for tgt_num in [500, 250, 100, 50, 25, 10, 1]:
        tgt_list = f"/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/9_TgtNum/target{tgt_num}.list"
        Target_vcf_path = outdir / f"target{tgt_num}.vcf.gz"
        subprocess.run(f"bcftools view -S {tgt_list} -Oz -o {Target_vcf_path} {Target_vcf_tmp}", shell=True)
        subprocess.run(f"bcftools index -t {Target_vcf_path}", shell=True)
        Target_vcf_tmp = Target_vcf_path
        
    Ref_vcf = indir / "ref.vcf.gz"
    ref_list = "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/9_TgtNum/ref.list"
    ref_map = "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/9_TgtNum/ref.map"
    ref_vcf_path = outdir / "ref.vcf.gz"
    subprocess.run(f"bcftools view -S {ref_list} -Oz -o {ref_vcf_path} {Ref_vcf}", shell=True)
    subprocess.run(f"bcftools index -t {ref_vcf_path}", shell=True)
    subprocess.run(f"cp {ref_map} {ref_map_path}", shell=True)    

    print(f"Done: {seed}")

if __name__ == "__main__":
    main()





        

