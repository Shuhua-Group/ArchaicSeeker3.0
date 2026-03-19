# import pandas as pd
# import os
# import sys
# import numpy as np
# from time import time

# list_file = "/home/linhuanyu/share1/20_AS3/results/inference/0_Scripts/1src/0_batch/full_1src_list.txt"

# # 读取任务列表
# with open(list_file, "r") as f:
#     lines = [line.strip() for line in f if line.strip()]

# def chrom_key(c):
#     # 自然排序：1..22,X,Y,MT
#     s = str(c).replace("chr","")
#     d = {"X": 23, "Y": 24, "MT": 25, "M": 25}
#     try:
#         return int(s)
#     except ValueError:
#         return d.get(s, 1_000_000)

# for idx, line in enumerate(lines, 1):
#     pop, nref, ntgt, seed = line.split()
#     dir_path   = f"/home/linhuanyu/share1/20_AS3/results/inference/Sprime/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}"
#     input_bed  = os.path.join(dir_path, "sprime.out.score")
#     output_bed = os.path.join(dir_path, "Infered_1src.bed")

#     t0 = time()
#     print(f"[{idx}/{len(lines)}] start {dir_path}", flush=True)

#     if not os.path.exists(input_bed):
#         print(f"  [skip] not found: {input_bed}", flush=True)
#         continue

#     try:
#         # 只读必要列，指定类型，避免 dtype 推断卡顿
#         usecols = ["CHROM","POS","SEGMENT","SCORE"]
#         df = pd.read_csv(
#             input_bed, sep=r"\s+", usecols=usecols, engine="c",
#             dtype={"CHROM":"string", "POS":"int64", "SEGMENT":"string", "SCORE":"float64"},
#             na_filter=False
#         )
#     except Exception as e:
#         print(f"  [error] read_csv: {e}", flush=True)
#         continue

#     if df.empty:
#         print("  [skip] empty file", flush=True)
#         continue

#     # 聚合一次得到片段起止（避免 transform 的重复计算）
#     try:
#         grp = (
#             df.groupby(["CHROM","SCORE","SEGMENT"], observed=True, sort=False)["POS"]
#               .agg(START_POS="min", END="max")
#               .reset_index()
#         )
#         # BED：0-based start（Sprime POS 通常是 1-based）
#         grp["START"] = (grp["START_POS"] - 1).astype("int64")
#         grp["END"]   = grp["END"].astype("int64")
#         out = grp[["CHROM","START","END","SEGMENT","SCORE"]].copy()
#     except Exception as e:
#         print(f"  [error] groupby/agg: {e}", flush=True)
#         continue

#     if out.empty:
#         print("  [skip] no segments after grouping", flush=True)
#         continue

#     # 排序：先按 CHROM 的自然顺序，再按 START、SEGMENT
#     out = out.sort_values(
#         by=["CHROM","START","SEGMENT"],
#         key=lambda s: s.map(chrom_key) if s.name=="CHROM" else s
#     ).reset_index(drop=True)
#     out = out[out["SCORE"] >= 150000]
#     out["SEGMENT"] = 0
#     out = out[["CHROM","START","END","SEGMENT"]]

#     try:
#         # to_csv 比 np.savetxt 更快、更稳
#         out.to_csv(output_bed, sep="\t", header=False, index=False)
#     except Exception as e:
#         print(f"  [error] write: {e}", flush=True)
#         continue

#     print(f"  [ok] wrote {output_bed} rows={len(out)} time={time()-t0:.1f}s", flush=True)

import pandas as pd
import os
import sys
import numpy as np
from time import time

list_file = "/home/linhuanyu/share1/20_AS3/0_Scripts/1src/0_batch/full_1src_list.txt"

# 读取任务列表
with open(list_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]


for idx, line in enumerate(lines, 1):
    pop, nref, ntgt, seed = line.split()
    dir_path   = f"/home/linhuanyu/share1/20_AS3/results/inference/Sprime/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}"
    input_rate  = os.path.join(dir_path, "sprime.out.match.rate")
    input_bed   = os.path.join(dir_path, "sprime.out.score")
    output_src_bed = os.path.join(dir_path, "Infered_1src_src.bed")
    output_src1_bed = os.path.join(dir_path, "Infered_1src_src1.bed")
    output_src2_bed = os.path.join(dir_path, "Infered_1src_src2.bed")

    score_df = pd.read_csv(input_bed, sep=r"\s+", header=0)
    rate_df  = pd.read_csv(input_rate, sep=r"\s+", header=0)

    df = pd.merge(rate_df, score_df, left_on="seg", right_on="SEGMENT", how="left")
    # df.to_csv(os.path.join(dir_path, "tmp.txt"), sep="\t", header=False, index=False)
    df = df[df["SCORE"] >= 150000]
    df = df[(df['src1'] + df['src2']) > 0]
    df['src'] = 'src'
    
    df.loc[df['src1'] > df[ 'src2'], 'src'] = 'src1'
    df.loc[df['src1'] < df['src2'], 'src'] = 'src2'
    df['index'] = 1

    df = df[['chr','from','to','src','index']]
    df.drop_duplicates(inplace=True)
    # df = df.iloc[1:].copy()
    # print(df.head())
    df_src = df[['chr','from','to','index']]
    df_src['Archaic'] = 1
    df_src.to_csv(output_src_bed, sep="\t", header=False, index=False)

    df_src1 = df[df['src'] == 'src1']
    df_src1 = df_src1[['chr','from','to','index']]
    df_src1.to_csv(output_src1_bed, sep="\t", header=False, index=False)

    df_src2 = df[df['src'] == 'src2']
    df_src2 = df_src2[['chr','from','to','index']]
    df_src2.to_csv(output_src2_bed, sep="\t", header=False, index=False)

    print(f"[{idx}/{len(lines)}] {pop} {nref} {ntgt} {seed} done")