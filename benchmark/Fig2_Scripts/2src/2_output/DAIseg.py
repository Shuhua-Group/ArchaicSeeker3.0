import pandas as pd
import os
import re

list_file = "/home/linhuanyu/share1/20_AS3/0_Scripts/2src/0_batch/full_2src_list.txt"

# 现在 decode 文件名是 tsk_xxx.decode.txt，没有 hap 号在文件名中
pattern = re.compile(r"tsk_(\d+)\.decode.txt$")

with open(list_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

for line in lines:
    pop, nref, ntgt, seed = line.split()
    dir_path = f"/home/linhuanyu/share1/20_AS3/results/inference/DAIseg/simulated_result/{pop}/nref_{nref}/ntgt_{ntgt}/{seed}"
    decode_dir = os.path.join(dir_path, "decode")
    output_bed = os.path.join(dir_path, "Infered_2src.bed")

    if not os.path.isdir(dir_path):
        print(f"[skip] not a directory: {dir_path}")
        continue
    if not os.path.isdir(decode_dir):
        print(f"[skip] no decode dir: {decode_dir}")
        continue

    # 只要 tsk_xxx.decode.txt
    names = [fn for fn in os.listdir(decode_dir) if pattern.match(fn)]
    if not names:
        print(f"[skip] no decoded files in {decode_dir}")
        continue

    # 按 tsk 里的数字排序（group(1)）
    names.sort(key=lambda fn: int(pattern.match(fn).group(1)))

    dfs = []
    # 给每个 tsk 编一个 index，下面再 ×2 + hap 来区分 ID
    name2idx = {fn: i for i, fn in enumerate(names)}

    for fn in names:
        m = pattern.match(fn)
        tsk_num = int(m.group(1))   # tsk 编号
        full_path = os.path.join(decode_dir, fn)

        try:
            # 你的文件看起来是空格/Tab 分隔，有表头
            df = pd.read_csv(full_path, sep=r"\s+", header=0)
        except Exception as e:
            print(f"[warn] failed to read {full_path}: {e}")
            continue

        # 基本列检查一下
        for col in ["chrom", "start", "end"]:
            if col not in df.columns:
                print(f"[warn] {full_path} missing column '{col}', skip this file.")
                df = None
                break
        if df is None:
            continue

        # 找到 start == 0 的行索引，用来切 hap 段
        zero_idx = df.index[df["start"] == 0].tolist()
        if len(zero_idx) == 0:
            print(f"[warn] {full_path} has no start == 0, treat as single hap.")
            zero_idx = [df.index[0]]
        elif len(zero_idx) == 1:
            # 只有一个 start=0，就当只有 hap1
            pass
        elif len(zero_idx) >= 2:
            # 多于 2 个也先只用前两个（正常应该是 2 个）
            if len(zero_idx) > 2:
                print(f"[warn] {full_path} has {len(zero_idx)} start==0, using first 2.")
            zero_idx = zero_idx[:2]

        # 构造每个 hap 的切片区域
        hap_segments = []
        if len(zero_idx) == 1:
            # 只有一段：hap 1
            hap_segments.append((1, df.index[0], df.index[-1] + 1))
        else:
            i1, i2 = zero_idx
            # hap1：从第一段 0 开始，到第二段 0 之前
            hap_segments.append((0, i1, i2))
            # hap2：从第二段 0 到结尾
            hap_segments.append((1, i2, df.index[-1] + 1))

        for hap_num, start_idx, end_idx in hap_segments:
            df_hap = df.iloc[start_idx:end_idx].copy()

            # 过滤出 archaic 段：state == 'Den1'
            if "state" in df_hap.columns:
                df_sel = df_hap[(df_hap["state"] == "Neanderthal") | (df_hap["state"] == "Den1") | (df_hap["state"] == "Ghost")| (df_hap["state"] == "Archaic")| (df_hap["state"] == "Denisovan")| (df_hap["state"] == "Bonobo")].copy()
            else:
                # 没有 state 就全保留
                df_sel = df_hap.copy()

            if df_sel.empty:
                continue

            # 只保留 chrom / start / end
            df_sel = df_sel[["chrom", "start", "end"]].copy()

            # 构造一个唯一 ID：每个 tsk 有两个 hap
            # 例如：tsk_idx=0 → hap1: ID=1, hap2: ID=2; tsk_idx=1 → 3,4 ...
            tsk_idx = name2idx[fn]
            df_sel["ID"] = tsk_idx * 2 + hap_num
            df_sel["Archaic"] = 1

            dfs.append(df_sel)

    if not dfs:
        print(f"[skip] no rows after filtering in {dir_path}")
        continue

    out_df = pd.concat(dfs, ignore_index=True)

    # 写 bed：chrom  start  end  ID  Archaic
    out_df.to_csv(output_bed, sep="\t", header=False, index=False)
    print(f"[ok] wrote {output_bed} ({len(out_df)} rows)")







