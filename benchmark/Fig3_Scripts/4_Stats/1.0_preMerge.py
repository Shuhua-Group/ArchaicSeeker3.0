#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# 导入 accuracy 函数（目前脚本里没用到，但先保留）
# =====================================================================

ACC_PATH = "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/acc.py"
ACC_DIR = os.path.dirname(ACC_PATH)
sys.path.append(ACC_DIR)
from acc import cal_accuracy_hap  # noqa: F401  # 预留


# =====================================================================
# 参数解析
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Task for ArchaicSeeker3")
    parser.add_argument("--seed", type=str, required=True, help="seed")
    parser.add_argument("--prefix", type=str, required=True, help="prefix")
    return parser.parse_args()


# =====================================================================
# 1) 模拟真值 & 推断结果逐 hap 匹配，保留 overlap>0 的所有记录
# =====================================================================

def get_best_matches(
    infered_bed: Path,
    sim_info_bed: Path,
    out_dir: Path,
    prefix: str
) -> pd.DataFrame:
    """
    对每个 hap_id，将模拟片段(sim_info)和推断片段(infered)做笛卡尔积，
    计算 overlap（kb）与 overlap_ratio（相对 Sim_Length），
    并保留所有 overlap>0 的记录。

    返回：包含 overlap>0 的 DataFrame。
    """

    # ---- 读入推断结果 ----
    infered_df = pd.read_csv(infered_bed, sep="\t", header=None)
    # 0,1,2,3,4,6 -> chr, start, end, hap_id, Archaic, Score
    infered_df = infered_df[[0, 1, 2, 3, 4, 6]].copy()
    infered_df.columns = ["chr", "start", "end", "hap_id", "Archaic", "Score"]

    # 长度 kb
    infered_df["Infered_Length"] = (infered_df["end"] - infered_df["start"]) / 1000.0
    infered_df = infered_df.reset_index().rename(columns={"index": "infer_id"})

    # ---- 读入模拟信息 ----
    sim_info_df = pd.read_csv(sim_info_bed, sep="\t", header=None)
    sim_info_df = sim_info_df[[0, 1, 2, 3, 4, 5]].copy()
    sim_info_df.columns = ["chr", "start", "end", "hap_id", "Archaic", "Sim_Length"]
    sim_info_df = sim_info_df.reset_index().rename(columns={"index": "id"})

    # ---- merge by hap_id ----
    siminfered = sim_info_df.merge(
        infered_df,
        on="hap_id",
        how="left",
        suffixes=("_sim", "_infered"),
    )

    if siminfered.empty:
        print(f"[WARN] siminfered is empty after merge for {prefix}", file=sys.stderr)

    # ---- 计算 overlap (kb) & overlap_ratio ----
    start_max = np.maximum(siminfered["start_sim"], siminfered["start_infered"])
    end_min = np.minimum(siminfered["end_sim"], siminfered["end_infered"])

    siminfered["overlap"] = (end_min - start_max).clip(lower=0) / 1000.0
    siminfered["overlap_ratio"] = siminfered["overlap"] / siminfered["Sim_Length"]

    # 只保留 overlap > 0
    siminfered_nonzero = siminfered[siminfered["overlap"] > 0].copy()

    if siminfered_nonzero.empty:
        print(f"[WARN] No overlap>0 segments for {prefix}", file=sys.stderr)

    # 选取并重命名列
    cols = [
        "id", "chr_sim", "start_sim", "end_sim", "hap_id",
        "Sim_Length", "Archaic_sim", "Archaic_infered",
        "overlap_ratio", "overlap",
        "start_infered", "end_infered", "Infered_Length", "Score",
    ]
    siminfered_nonzero = siminfered_nonzero[cols].copy()
    siminfered_nonzero.columns = [
        "id", "chr", "start", "end", "hap_id",
        "Sim_Length", "Archaic_sim", "Archaic_infered",
        "overlap_ratio", "overlap",
        "start_infered", "end_infered", "Infered_Length", "Score",
    ]

    # 输出
    out_path = out_dir / f"siminfered.overlapnonzero.{prefix}.bed"
    siminfered_nonzero.to_csv(out_path, sep="\t", index=False)
    print(f"[INFO] Saved all overlap>0 matches to: {out_path}")

    return siminfered_nonzero


# =====================================================================
# 2) 对具有多个 infer 片段的同一模拟片段统计“gap”情况
# =====================================================================

def summarize_gaps_by_hap(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入 df 必须包含：
        id, hap_id, start, end, Sim_Length, start_infered, end_infered, ...

    对于每个 id（一个模拟片段）：
        - 如果该 id 只匹配到 1 条 infer，则不统计；
        - 如果该 id 匹配到多条 infer，则：
            * 将 infer 片段按 start_infered 排序；
            * 计算相邻 infer 之间的 gap（中间没覆盖的部分）；
            * 得到：
                - infer_start / infer_end（合并覆盖范围）
                - max_gap_kb
                - total_gap_kb

    返回 summary DataFrame：
        id, hap_id, sim_start, sim_end, Sim_Length,
        n_infer, infer_start, infer_end, max_gap_kb, total_gap_kb
    """

    if df.empty:
        print("[WARN] Input df is empty in summarize_gaps_by_hap", file=sys.stderr)
        return pd.DataFrame(
            columns=[
                "id", "hap_id", "sim_start", "sim_end", "Sim_Length",
                "n_infer", "infer_start", "infer_end", "max_gap_kb", "total_gap_kb",
            ]
        )

    # 只保留 id 非唯一（说明有多个 infer 覆盖同一 sim）
    df_multi = df[df.duplicated("id", keep=False)].copy()
    if df_multi.empty:
        print("[INFO] No duplicated id in df; no multi-infer segments.", file=sys.stderr)
        return pd.DataFrame(
            columns=[
                "id", "hap_id", "sim_start", "sim_end", "Sim_Length",
                "n_infer", "infer_start", "infer_end", "max_gap_kb", "total_gap_kb",
            ]
        )

    results = []

    for gid, g in df_multi.groupby("id"):
        hap_id = g["hap_id"].iloc[0]
        sim_start = g["start"].iloc[0]
        sim_end = g["end"].iloc[0]
        sim_len = g["Sim_Length"].iloc[0]

        # 按 infer 坐标排序
        g_sorted = g.sort_values("start_infered")

        gap_lengths = []
        prev_end = None

        for _, row in g_sorted.iterrows():
            s = row["start_infered"]
            e = row["end_infered"]

            if prev_end is not None:
                gap_start = prev_end
                gap_end = s
                if gap_end > gap_start:
                    gap_lengths.append(gap_end - gap_start)

            prev_end = e if prev_end is None else max(prev_end, e)

        max_gap_kb = max(gap_lengths) / 1000.0 if gap_lengths else 0.0
        total_gap_kb = sum(gap_lengths) / 1000.0 if gap_lengths else 0.0

        infer_start = g_sorted["start_infered"].min()
        infer_end = g_sorted["end_infered"].max()

        results.append(
            {
                "id": gid,
                "hap_id": hap_id,
                "sim_start": sim_start,
                "sim_end": sim_end,
                "Sim_Length": sim_len,
                "n_infer": len(g_sorted),
                "infer_start": infer_start,
                "infer_end": infer_end,
                "max_gap_kb": max_gap_kb,
                "total_gap_kb": total_gap_kb,
            }
        )

    summary = pd.DataFrame(results)
    return summary


# =====================================================================
# 3) （备用）向 summary 追加原始 infer 中未覆盖的片段，并计算 min_dist_same_hap
# =====================================================================

def append_non_overlaps_and_update_dist(
    summary_df: pd.DataFrame,
    infered_bed: Path,
) -> pd.DataFrame:
    """
    summary_df: 已有片段 DataFrame，至少包含列 ["hap_id", "start", "end"]
    infered_bed: 推断片段的 bed 文件路径（列: chr, start, end, hap_id, ...）

    逻辑：
      1）从 infered_bed 读入 start/end/hap_id
      2）如果某条 infer 片段与 summary 中同 hap_id 的任何片段 overlap，则跳过
      3）否则将该 infer 片段追加到 summary_df
      4）在合并后的 summary_df 上按 hap_id 计算 min_dist_same_hap（基于 start/end）
    返回：
      更新后的 summary_df
    """

    if summary_df.empty:
        print("[WARN] summary_df is empty in append_non_overlaps_and_update_dist", file=sys.stderr)
        return summary_df

    # ---- 读入 infered ----
    infered_df = pd.read_csv(infered_bed, sep="\t", header=None)
    # 假定第 1,2,3 列分别是 start, end, hap_id
    infered_df = infered_df[[1, 2, 3]].copy()
    infered_df.columns = ["start", "end", "hap_id"]

    # 结构对齐
    summary_df = summary_df[["hap_id", "start", "end"]].copy()

    # 转成数值型，避免类型问题
    summary_df["start"] = summary_df["start"].astype(int)
    summary_df["end"] = summary_df["end"].astype(int)
    infered_df["start"] = infered_df["start"].astype(int)
    infered_df["end"] = infered_df["end"].astype(int)

    new_rows = []

    grouped = summary_df.groupby("hap_id")

    for _, row in infered_df.iterrows():
        h = row["hap_id"]
        s = row["start"]
        e = row["end"]

        has_overlap = False
        if h in grouped.groups:
            g = grouped.get_group(h)
            no_overlap_mask = (e < g["start"]) | (s > g["end"])
            has_overlap = not no_overlap_mask.all()

        if has_overlap:
            continue  # 与已有 summary 片段重叠 -> 跳过

        new_rows.append(row)

    if new_rows:
        summary_df = pd.concat(
            [summary_df, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

    # ---- 重新按 hap_id 计算 min_dist_same_hap ----
    summary_df = summary_df.sort_values(["hap_id", "start"]).reset_index(drop=True)
    summary_df["min_dist_same_hap"] = np.nan

    for hap, g in summary_df.groupby("hap_id"):
        if len(g) == 1:
            continue

        starts = g["start"].to_numpy(dtype=float)
        ends = g["end"].to_numpy(dtype=float)

        prev_end = np.roll(ends, 1)
        prev_end[0] = np.nan
        dist_prev = starts - prev_end

        next_start = np.roll(starts, -1)
        next_start[-1] = np.nan
        dist_next = next_start - ends

        dist_prev = np.where(
            np.isnan(dist_prev), np.nan, np.where(dist_prev < 0, 0, dist_prev)
        )
        dist_next = np.where(
            np.isnan(dist_next), np.nan, np.where(dist_next < 0, 0, dist_next)
        )

        min_dist = np.nanmin(np.vstack([dist_prev, dist_next]), axis=0)
        summary_df.loc[g.index, "min_dist_same_hap"] = min_dist

    return summary_df


# =====================================================================
# main
# =====================================================================

def main() -> None:
    args = parse_args()
    seed = args.seed
    prefix = args.prefix

    basedir = Path(
        "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut"
    )
    out_dir = basedir / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] seed:   {seed}")
    print(f"[INFO] prefix: {prefix}")
    print(f"[INFO] basedir: {basedir}")
    print(f"[INFO] out_dir: {out_dir}")

    Archaic_sim_info_bed = out_dir / "Archaic.sim.info.bed"
    Archaic_infered_bed_score = out_dir / f"{prefix}.bed"

    # 1) 模拟-推断匹配，保留 overlap>0
    siminfered_nonzero = get_best_matches(
        Archaic_infered_bed_score,
        Archaic_sim_info_bed,
        out_dir,
        prefix,
    )

    # 2) 统计 gap
    gaps_df = summarize_gaps_by_hap(siminfered_nonzero)
    gaps_out = out_dir / f"gaps.summary.{prefix}.txt"
    gaps_df.to_csv(gaps_out, sep="\t", index=False)
    print(f"[INFO] Saved gaps summary to: {gaps_out}")

    # 若你之后要用 append_non_overlaps_and_update_dist，可以在这里接：
    summary_for_dist = gaps_df[["hap_id", "infer_start", "infer_end"]].rename(
        columns={"infer_start": "start", "infer_end": "end"}
    )
    summary_with_dist = append_non_overlaps_and_update_dist(
        summary_for_dist, Archaic_infered_bed_score
    )
    summary_with_dist.to_csv(out_dir / f"gaps_with_dist.{prefix}.txt", sep="\t", index=False)

    print("[INFO] Done!")


if __name__ == "__main__":
    main()
