import pandas as pd
import os

# ====== 输入文件列表 ======
list_file = "/home/linhuanyu/share1/20_AS3/0_Scripts/2src/0_batch/full_2src_list.txt"

# 读取每一行
with open(list_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# 遍历每一行
for line in lines:
    pop, nref, ntgt, seed = line.split()

    dir_path = f"/home/linhuanyu/share1/20_AS3/results/inference/IBDmix/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}"

    input_bed_1 = os.path.join(dir_path, "ibdmix_arc1_output.txt")
    input_bed_2 = os.path.join(dir_path, "ibdmix_arc2_output.txt")
    input_map = os.path.join(f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker3.0/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}", "hapmap.txt")
    output_src_bed = os.path.join(dir_path, "Infered_2src_src.bed")
    output_src1_bed = os.path.join(dir_path, "Infered_2src_src1.bed")
    output_src2_bed = os.path.join(dir_path, "Infered_2src_src2.bed")


    # 检查文件是否存在
    if not (os.path.exists(input_bed_1) and os.path.exists(input_bed_2) and os.path.exists(input_map)):
        print(f"[跳过] 缺文件: {dir_path}")
        continue

    # ====== 处理 ======

    def process_bed(input_bed):
        df = pd.read_csv(input_bed, sep=r"\s+", header=0)
        map_df = pd.read_csv(input_map, sep=r"\s+", header=None)

        # 去掉 copy 结尾的行
        df = df[~df['ID'].str.endswith('copy', na=False)]

        # 如果过滤后为空，直接返回结构正确的空表
        if df.empty:
            return pd.DataFrame(columns=['chrom', 'start', 'end', 'ID'])

        # 加后缀
        df['ID'] = df['ID'] + '_1'

        # 建映射字典（如果需要映射，这里先 map 再转 int）
        tsk_to_idx = {tsk_id: i for i, tsk_id in enumerate(map_df.iloc[:, 1])}
        df['ID'] = df['ID'].map(tsk_to_idx)

        # 再次判断是否空
        if df.empty:
            return pd.DataFrame(columns=['chrom', 'start', 'end', 'ID'])

        # 转 int 之前丢掉 NaN
        df = df.dropna(subset=['ID'])
        if df.empty:
            return pd.DataFrame(columns=['chrom', 'start', 'end', 'ID'])
        df['ID'] = df['ID'].astype(int)

        # 过滤 slod
        df = df[df['slod'] >= 4]
        if df.empty:
            return pd.DataFrame(columns=['chrom', 'start', 'end', 'ID'])

        # 输出需要的列
        df_out = df[['chrom', 'start', 'end', 'ID']].reset_index(drop=True)
        return df_out
    
    df_out_1 = process_bed(input_bed_1)
    df_out_2 = process_bed(input_bed_2)

    df_out_1.to_csv(output_src1_bed, sep="\t", header=False, index=False)
    df_out_2.to_csv(output_src2_bed, sep="\t", header=False, index=False)

    df_out_1['Archaic'] = 1
    df_out_2['Archaic'] = 2

    df_out = pd.concat([df_out_1, df_out_2])
    df_out.to_csv(output_src_bed, sep="\t", header=False, index=False)
                    

    print(f"[完成] {output_src_bed}")
    print(f"[完成] {output_src1_bed}")
    print(f"[完成] {output_src2_bed}")
