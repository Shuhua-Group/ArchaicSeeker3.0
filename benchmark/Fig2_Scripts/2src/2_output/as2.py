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

    dir_path = f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker2.0/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}"

    input_bed = os.path.join(dir_path, "archaicseeker2.out.seg")
    input_map = os.path.join(f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker3.0/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}", "hapmap.txt")
    output_src_bed = os.path.join(dir_path, "Infered_2src_src.bed")
    output_src1_bed = os.path.join(dir_path, "Infered_2src_src1.bed")
    output_src2_bed = os.path.join(dir_path, "Infered_2src_src2.bed")

    # 检查文件是否存在
    if not (os.path.exists(input_bed) and os.path.exists(input_map)):
        print(f"[跳过] 缺文件: {dir_path}")
        continue

    # ====== 处理 ======
    df = pd.read_csv(input_bed, sep=r"\s+", header=0)
    map_df = pd.read_csv(input_map, sep=r"\s+", header=None)
    tsk_to_idx = {tsk_id: i for i, tsk_id in enumerate(map_df.iloc[:, 1])}
    df['ID'] = df['ID'].map(tsk_to_idx)

    if pop == 'AS2_HumanNeanderthalDenisovan':
        df1 = df[df['BestMatchedPop'] == 'Neanderthal']
        df2 = df[df['BestMatchedPop'] == 'Denisovan']
        df_src = df[(df['BestMatchedPop'] == 'Denisovan_Neanderthal') | (df['BestMatchedPop'] == 'Neanderthal_Denisovan') | (df['BestMatchedPop'] == 'Denisovan') | (df['BestMatchedPop'] == 'Neanderthal')]
        df_src['Archaic'] = 3
        df_src.loc[df_src['BestMatchedPop'] == 'Neanderthal', 'Archaic'] = 1
        df_src.loc[df_src['BestMatchedPop'] == 'Denisovan', 'Archaic'] = 2
    elif pop == 'ChimpBonoboGhost':
        df1 = df[df['BestMatchedPop'] == 'Ghost']
        df2 = df[df['BestMatchedPop'] == 'Bonobo']
        df_src = df[(df['BestMatchedPop'] == 'Bonobo_Ghost') | (df['BestMatchedPop'] == 'Ghost_Bonobo') | (df['BestMatchedPop'] == 'Ghost') | (df['BestMatchedPop'] == 'Bonobo')]
        df_src['Archaic'] = 3
        df_src.loc[df_src['BestMatchedPop'] == 'Ghost', 'Archaic'] = 1
        df_src.loc[df_src['BestMatchedPop'] == 'Bonobo', 'Archaic'] = 2
    elif pop == 'HumanArchaic':
        df1 = df[df['BestMatchedPop'] == 'ArchaicAFR']
        df2 = df[df['BestMatchedPop'] == 'Neanderthal']
        df_src = df[(df['BestMatchedPop'] == 'ArchaicAFR_Neanderthal') | (df['BestMatchedPop'] == 'Neanderthal_ArchaicAFR') | (df['BestMatchedPop'] == 'ArchaicAFR') | (df['BestMatchedPop'] == 'Neanderthal')]
        df_src['Archaic'] = 3
        df_src.loc[df_src['BestMatchedPop'] == 'ArchaicAFR', 'Archaic'] = 1
        df_src.loc[df_src['BestMatchedPop'] == 'Neanderthal', 'Archaic'] = 2
    elif pop == 'HumanNeanderthalDenisovan':
        df1 = df[df['BestMatchedPop'] == 'Neanderthal']
        df2 = df[df['BestMatchedPop'] == 'Denisovan']
        df_src = df[(df['BestMatchedPop'] == 'Denisovan_Neanderthal') | (df['BestMatchedPop'] == 'Neanderthal_Denisovan') | (df['BestMatchedPop'] == 'Denisovan') | (df['BestMatchedPop'] == 'Neanderthal')]
        df_src['Archaic'] = 3
        df_src.loc[df_src['BestMatchedPop'] == 'Neanderthal', 'Archaic'] = 1
        df_src.loc[df_src['BestMatchedPop'] == 'Denisovan', 'Archaic'] = 2
    else:
        raise ValueError(f"未知的群体: {pop}")

    df1_out = df1[['Contig','Start(bp)','End(bp)','ID']]
    df2_out = df2[['Contig','Start(bp)','End(bp)','ID']]
    df_src_out = df_src[['Contig','Start(bp)','End(bp)','ID','Archaic']]

    df1_out.to_csv(output_src1_bed, sep="\t", header=False, index=False)
    df2_out.to_csv(output_src2_bed, sep="\t", header=False, index=False)
    df_src_out.to_csv(output_src_bed, sep="\t", header=False, index=False)

    print(f"[完成] {output_src_bed}")
    print(f"[完成] {output_src1_bed}")
    print(f"[完成] {output_src2_bed}")
