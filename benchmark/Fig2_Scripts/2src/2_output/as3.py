import pandas as pd
import os

# ====== 输入文件列表 ======
list_file = "/home/linhuanyu/share1/20_AS3/results/inference/0_Scripts/2src/0_batch/full_2src_list.txt"

# 读取每一行
with open(list_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# 遍历每一行
for line in lines:
    pop, nref, ntgt, seed = line.split()

    dir_path = f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker3.0/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}"

    input_bed = os.path.join(dir_path, "AS3_Mamba_Smoother_Aug4.bed")
    input_map = os.path.join(dir_path, "hapmap.txt")
    output_src_bed = os.path.join(dir_path, "Infered_2src_src.bed")
    output_src1_bed = os.path.join(dir_path, "Infered_2src_src1.bed")
    output_src2_bed = os.path.join(dir_path, "Infered_2src_src2.bed")

    # 检查文件是否存在
    if not (os.path.exists(input_bed) and os.path.exists(input_map)):
        print(f"[跳过] 缺文件: {dir_path}")
        continue

    # ====== 处理 ======
    df = pd.read_csv(input_bed, sep=r"\s+", header=None)
    map_df = pd.read_csv(input_map, sep=r"\s+", header=None)
    tsk_to_idx = {tsk_id: i for i, tsk_id in enumerate(map_df.iloc[:, 1])}
    df[3] = df[3].map(tsk_to_idx)
    df = df[df[6] >= 0.4]

    df_src = df[[0, 1, 2, 3]]

    if pop == 'AS2_HumanNeanderthalDenisovan':
        df1 = df[df[4] == 2]
        df2 = df[df[4] == 1]
    elif pop == 'ChimpBonoboGhost':
        df1 = df[df[4] == 2]
        df2 = df[df[4] == 1]
    elif pop == 'HumanArchaic':
        df1 = df[df[4] == 1]
        df2 = df[df[4] == 2]
    elif pop == 'HumanNeanderthalDenisovan':
        df1 = df[df[4] == 2]
        df2 = df[df[4] == 1]
    else:
        raise ValueError(f"未知的群体: {pop}")

    df_src1 = df1[[0, 1, 2, 3]]
    df_src2 = df2[[0, 1, 2, 3]]

    df_src.to_csv(output_src_bed, sep="\t", header=False, index=False)
    df_src1.to_csv(output_src1_bed, sep="\t", header=False, index=False)
    df_src2.to_csv(output_src2_bed, sep="\t", header=False, index=False)

    print(f"[完成] {output_src_bed}")
    print(f"[完成] {output_src1_bed}")
    print(f"[完成] {output_src2_bed}")
