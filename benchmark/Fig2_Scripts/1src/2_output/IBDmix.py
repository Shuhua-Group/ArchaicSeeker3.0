import pandas as pd
import os

# ====== 输入文件列表 ======
list_file = "/home/linhuanyu/share1/20_AS3/0_Scripts/1src/0_batch/full_1src_list.txt"

# 读取每一行
with open(list_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# 遍历每一行
for line in lines:
    pop, nref, ntgt, seed = line.split()

    dir_path = f"/home/linhuanyu/share1/20_AS3/results/inference/IBDmix/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}"

    input_bed = os.path.join(dir_path, "ibdmix_output.txt")
    input_map = os.path.join(f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker3.0/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}", "hapmap.txt")
    output_bed = os.path.join(dir_path, "Infered_1src.bed")

    # 检查文件是否存在
    if not (os.path.exists(input_bed) and os.path.exists(input_map)):
        print(f"[跳过] 缺文件: {dir_path}")
        continue

    # ====== 处理 ======
    df = pd.read_csv(input_bed, sep=r"\s+", header=0)
    map_df = pd.read_csv(input_map, sep=r"\s+", header=None)
    df = df[~df['ID'].str.endswith('copy')]
    df['ID'] = df['ID'] + '_1'
    tsk_to_idx = {tsk_id: i for i, tsk_id in enumerate(map_df.iloc[:, 1])}
    df['ID'] = df['ID'].map(tsk_to_idx)
    df = df[df['slod'] >= 4]
    df_out = df[['chrom','start','end','ID']]
    df_out['Archaic'] = 1
    df_out.to_csv(output_bed, sep="\t", header=False, index=False)

    print(f"[完成] {output_bed}")
