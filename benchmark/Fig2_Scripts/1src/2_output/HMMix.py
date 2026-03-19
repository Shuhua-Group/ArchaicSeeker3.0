import pandas as pd
import os
import re

list_file = "/home/linhuanyu/share1/20_AS3/0_Scripts/1src/0_batch/full_1src_list.txt"

pattern = re.compile(r"tsk_(\d+)\.decoded\.hap(\d)\.txt$")

with open(list_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

for line in lines:
    pop, nref, ntgt, seed = line.split()
    dir_path = f"/home/linhuanyu/share1/20_AS3/results/inference/HMMix/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}"
    output_bed = os.path.join(dir_path, "Infered_1src.bed")

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
            last_col = df.columns[-1]
            df_sel = df[(df['state'] == 'Archaic') & (df[last_col] != 0)]
        if df_sel.empty:
            continue

        # 组装 ID（如需数值 ID，可改成 f"{global_index}" 或映射）
        df_sel = df_sel[['chrom', 'start', 'end']].copy()
        df_sel['ID'] = name2idx[fn]   # or use just tsk_num, or a numeric mapping
        df_sel['Archaic'] = 1
        dfs.append(df_sel)

    if not dfs:
        print(f"[skip] no rows after filtering in {dir_path}")
        continue

    out_df = pd.concat(dfs, ignore_index=True)
    out_df.to_csv(output_bed, sep="\t", header=False, index=False)
    print(f"[ok] wrote {output_bed} ({len(out_df)} rows)")






