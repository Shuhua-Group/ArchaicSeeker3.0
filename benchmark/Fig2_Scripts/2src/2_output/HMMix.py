import pandas as pd
import os
import re

list_file = "/home/linhuanyu/share1/20_AS3/0_Scripts/2src/0_batch/full_2src_list.txt"

pattern = re.compile(r"tsk_(\d+)\.decoded\.hap(\d)\.txt$")

with open(list_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

for line in lines:
    pop, nref, ntgt, seed = line.split()
    dir_path = f"/home/linhuanyu/share1/20_AS3/results/inference/HMMix/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}"
    output_src_bed = os.path.join(dir_path, "Infered_2src_src.bed")
    output_src1_bed = os.path.join(dir_path, "Infered_2src_src1.bed")
    output_src2_bed = os.path.join(dir_path, "Infered_2src_src2.bed")

    if not os.path.isdir(dir_path):
        print(f"[skip] not a directory: {dir_path}")
        continue

    # 仅选择 tsk_x.decoded.hapy.txt，并按 tsk 数字、hap 数字排序
    names = [fn for fn in os.listdir(dir_path) if pattern.match(fn)]
    if not names:
        print(f"[skip] no decoded files in {dir_path}")
        continue
    names.sort(key=lambda fn: (int(pattern.match(fn).group(1)),
                               int(pattern.match(fn).group(2))))

    dfs = []
    name2idx = {fn: i for i, fn in enumerate(names)}
    for fn in names:
        m = pattern.match(fn)
        tsk_num = int(m.group(1))
        hap_num = int(m.group(2))
        full_path = os.path.join(dir_path, fn)

        try:
            df = pd.read_csv(full_path, sep=r"\s+", header=0)
        except Exception as e:
            print(f"[warn] failed to read {full_path}: {e}")
            continue

        # 过滤“Archaic”状态（确认列名与拼写）
        if 'state' not in df.columns:
            print(f"[warn] no 'state' column in {full_path}, keep all rows")
            df_sel = df
        else:
            last_two_cols = df.columns[-2:]
            df.rename(columns={last_two_cols[0]: 'src1', last_two_cols[1]: 'src2'}, inplace=True)
            df_sel = df[(df['state'] == 'Archaic') & (df['src1'] + df['src2'] != 0)]
            df_sel.loc[df_sel['src1'] > df_sel['src2'], 'state'] = 'src1'
            df_sel.loc[df_sel['src2'] > df_sel['src1'], 'state'] = 'src2'
            df_sel.loc[df_sel['src1'] == df_sel['src2'], 'state'] = 'src'
        if df_sel.empty:
            continue

        # 组装 ID（如需数值 ID，可改成 f"{global_index}" 或映射）
        df_sel = df_sel[['chrom', 'start', 'end','state']].copy()
        df_sel['ID'] = name2idx[fn]   # or use just tsk_num, or a numeric mapping
        dfs.append(df_sel)

    if not dfs:
        print(f"[skip] no rows after filtering in {dir_path}")
        continue

    out_df = pd.concat(dfs, ignore_index=True)
    out_df_src1 = out_df[out_df['state'] == 'src1']
    out_df_src1 = out_df_src1[['chrom', 'start', 'end','ID']]
    out_df_src2 = out_df[out_df['state'] == 'src2']
    out_df_src2 = out_df_src2[['chrom', 'start', 'end','ID']]
    out_df_src = out_df[(out_df['state'] == 'src1') | (out_df['state'] == 'src2') | (out_df['state'] == 'src')]
    out_df_src['Archaic'] = 3
    out_df_src.loc[out_df_src['state'] == 'src1', 'Archaic'] = 1
    out_df_src.loc[out_df_src['state'] == 'src2', 'Archaic'] = 2
    out_df_src = out_df_src[['chrom', 'start', 'end','ID','Archaic']]

    out_df_src.to_csv(output_src_bed, sep="\t", header=False, index=False)
    out_df_src1.to_csv(output_src1_bed, sep="\t", header=False, index=False)
    out_df_src2.to_csv(output_src2_bed, sep="\t", header=False, index=False)

    print(f"[ok] wrote {output_src1_bed} ({len(out_df_src1)} rows)")
    print(f"[ok] wrote {output_src2_bed} ({len(out_df_src2)} rows)")
    print(f"[ok] wrote {output_src_bed} ({len(out_df_src)} rows)")






