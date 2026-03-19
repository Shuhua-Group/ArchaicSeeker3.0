import pandas as pd
import numpy as np
import argparse
import os
import subprocess
from pathlib import Path

def merge_Den_bed(Den1_bed_path, Den2_bed_path, Den_bed_path):
    Den1_bed_raw = pd.read_csv(Den1_bed_path, sep="\t", header=None)
    Den2_bed_raw = pd.read_csv(Den2_bed_path, sep="\t", header=None)
    Den_bed_raw = pd.concat([Den1_bed_raw, Den2_bed_raw])
    Den_bed_raw.to_csv(Den_bed_path, sep="\t", header=False, index=False)


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

# def cut_Archaic_bed(archaic_bed_info_path, Length_bins = [0,5,10,15,20,25,30,50,75,100,100000]):
#     dir_name = os.path.dirname(archaic_bed_info_path)
#     df = pd.read_csv(archaic_bed_info_path, sep="\t", header=None)
#     df["Length_Bin"] = pd.cut(df[5], bins=Length_bins, labels=Length_bins[1:], right=False)
#     dfs = []
#     for i in df["Length_Bin"].unique():
#         dfs.append(df[df["Length_Bin"] == i])
#     for i in dfs:
#         i.to_csv(f"{dir_name}/Archaic.Simulated.Max{i['Length_Bin'].iloc[0]}kb.info", sep="\t", header=False, index=False)
#         i[[0,1,2,3]].to_csv(f"{dir_name}/Archaic.Simulated.Max{i['Length_Bin'].iloc[0]}kb.bed", sep="\t", header=False, index=False)
#         i.loc[i[4] == 'Den',[0,1,2,3]].to_csv(f"{dir_name}/Den.Simulated.Max{i['Length_Bin'].iloc[0]}kb.bed", sep="\t", header=False, index=False)
#         i.loc[i[4] == 'Nean',[0,1,2,3]].to_csv(f"{dir_name}/Nean.Simulated.Max{i['Length_Bin'].iloc[0]}kb.bed", sep="\t", header=False, index=False)
        
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Task for ArchaicSeeker3")
    parser.add_argument("--seed", type=str, required=True, help="seed")
    return parser.parse_args()

def main():
    args = parse_args()
    seed = args.seed
    print(f"seed: {seed}")

    # 输入文件夹
    indir = Path(f"/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut/{seed}")
    # 输出文件夹
    outdir = Path(f"/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut/{seed}")

    # 创建输出文件夹
    os.makedirs(outdir, exist_ok=True)

    # 原始模拟数据
    Target_vcf = indir / "target.vcf.gz"
    Ref_vcf = indir / "ref.vcf.gz"
    Ref_map = indir / "ref.map"

    # 原始真值
    Den1_bed_path = indir / "Den1.sim.bed"
    Den2_bed_path = indir / "Den2.sim.bed"
    Den_bed_path = indir / "Den.sim.bed"
    Nean_bed_path = indir / "Nean.sim.bed"
    Archaic_bed_path = indir / "Archaic.sim.bed"
    Archaic_bed_info_path = indir / "Archaic.sim.info.bed"

    merge_Den_bed(Den1_bed_path, Den2_bed_path, Den_bed_path)
    process_Archaic_bed(Den_bed_path, Nean_bed_path, Archaic_bed_path, Archaic_bed_info_path)

    # # 输出文件
    # Target_vcf_path = outdir / "Target.vcf.gz"
    # Ref_vcf_path = outdir / "Ref.vcf.gz"
    # Ref_map_path = outdir / "Ref.map"
    # Den_bed_path = outdir / "Den.Simulated.bed"
    # Nean_bed_path = outdir / "Nean.Simulated.bed"
    # Archaic_bed_path = outdir / "Archaic.Simulated.bed"
    # Archaic_bed_info_path = outdir / "Archaic.Simulated.info"

    # # 数据处理
    # subprocess.run(f"cp {Target_vcf_raw} {Target_vcf_path}", shell=True)
    # subprocess.run(f"bcftools index -t {Target_vcf}", shell=True)
    # subprocess.run(f"cp {Ref_vcf_raw} {Ref_vcf_path}", shell=True)
    # subprocess.run(f"bcftools index -t {Ref_vcf}", shell=True)
    # subprocess.run(f"cp {Ref_map_raw} {Ref_map_path}", shell=True)
    # subprocess.run(f"cp {Den_bed_raw} {Den_bed_path}", shell=True)
    # subprocess.run(f"cp {Nean_bed_raw} {Nean_bed_path}", shell=True)

    # process_Archaic_bed(Den_bed_raw, Nean_bed_raw, Archaic_bed_path, Archaic_bed_info_path)
    # cut_Archaic_bed(Archaic_bed_info_path)
    print(f"Done: {seed}")

if __name__ == "__main__":
    main()
