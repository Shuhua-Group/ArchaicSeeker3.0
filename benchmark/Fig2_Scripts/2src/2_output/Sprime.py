import pandas as pd
import os
import sys
import numpy as np
from time import time

list_file = "/home/linhuanyu/share1/20_AS3/0_Scripts/2src/0_batch/full_2src_list.txt"

# 读取任务列表
with open(list_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]


for idx, line in enumerate(lines, 1):
    pop, nref, ntgt, seed = line.split()
    dir_path   = f"/home/linhuanyu/share1/20_AS3/results/inference/Sprime/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}"
    input_rate  = os.path.join(dir_path, "sprime.2src.out.match.rate")
    input_bed   = os.path.join(dir_path, "sprime.2src.out.score")
    output_src_bed = os.path.join(dir_path, "Infered_2src_src.bed")
    output_src1_bed = os.path.join(dir_path, "Infered_2src_src1.bed")
    output_src2_bed = os.path.join(dir_path, "Infered_2src_src2.bed")

    score_df = pd.read_csv(input_bed, sep=r"\s+", header=0)
    rate_df  = pd.read_csv(input_rate, sep=r"\s+", header=0)

    df = pd.merge(rate_df, score_df, left_on="seg", right_on="SEGMENT", how="left")
    # df.to_csv(os.path.join(dir_path, "tmp.txt"), sep="\t", header=False, index=False)
    df = df[df["SCORE"] >= 150000]
    df['src'] = 3
    df.loc[df['src1'] > df[ 'src2'], 'src'] = 1
    df.loc[df['src1'] < df['src2'], 'src'] = 2
    df['index'] = 1

    df = df[['chr','from','to','index','src']]
    df.drop_duplicates(inplace=True)
    df_src = df[['chr','from','to','index','src']]
    # df_src['Archaic'] = 3
    df_src.to_csv(output_src_bed, sep="\t", header=False, index=False)

    df_src1 = df[df['src'] == 1]
    df_src1 = df_src1[['chr','from','to','index','src']]
    df_src1.to_csv(output_src1_bed, sep="\t", header=False, index=False)

    df_src2 = df[df['src'] == 2]
    df_src2 = df_src2[['chr','from','to','index','src']]
    df_src2.to_csv(output_src2_bed, sep="\t", header=False, index=False)

    print(f"[{idx}/{len(lines)}] {pop} {nref} {ntgt} {seed} done")