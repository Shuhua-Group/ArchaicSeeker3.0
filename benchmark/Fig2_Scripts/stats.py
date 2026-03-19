#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

ACC_PATH = "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/acc.py"
ACC_DIR = os.path.dirname(ACC_PATH)
sys.path.append(ACC_DIR)
from acc import cal_accuracy_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Task for ArchaicSeeker3")
    parser.add_argument("--demo", type=str, required=True, help="demo")
    parser.add_argument("--nref", type=int, required=True, help="nref")
    parser.add_argument("--ntgt", type=int, required=True, help="ntgt")
    parser.add_argument("--seed", type=str, required=True, help="seed")    
    return parser.parse_args()

# def Filter(AS3_raw_bed,AS3_filtered_bed,min_length,min_score):
#     df = pd.read_csv(AS3_raw_bed, sep="\t", header=None)
#     df = df[df[8] >= min_score]
#     df = df[df[2] - df[1] >= min_length * 1000]
#     df.to_csv(AS3_filtered_bed, sep="\t", header=None, index=False)
#     return df

# def Archaic2DenNean(Archaic_infered_bed ,Archaic_infered_tmp_bed,Den_infered_bed, Nean_infered_bed):
#     df = pd.read_csv(Archaic_infered_bed, sep="\t", header=None)
#     df[df[6] == 1][[0,1,2,5,6,7,8]].to_csv(Den_infered_bed, sep="\t", header=None, index=False)
#     df[df[6] == 2][[0,1,2,5,6,7,8]].to_csv(Nean_infered_bed, sep="\t", header=None, index=False)
#     df[[0,1,2,5,6,7,8]].to_csv(Archaic_infered_tmp_bed,sep='\t',header=None,index=False)

def Filter(AS3_raw_bed,AS3_filtered_bed,min_length,min_score):
    df = pd.read_csv(AS3_raw_bed, sep="\t", header=None)
    df = df[df[6] >= min_score]
    df = df[df[2] - df[1] >= min_length * 1000]
    df.to_csv(AS3_filtered_bed, sep="\t", header=None, index=False)
    return df

def Archaic2DenNean(Archaic_infered_bed ,Archaic_infered_tmp_bed,Den_infered_bed, Nean_infered_bed):
    df = pd.read_csv(Archaic_infered_bed, sep="\t", header=None)
    df[df[4] == 1].to_csv(Den_infered_bed, sep="\t", header=None, index=False)
    df[df[4] == 2].to_csv(Nean_infered_bed, sep="\t", header=None, index=False)
    df.to_csv(Archaic_infered_tmp_bed,sep='\t',header=None,index=False)

def process_Archaic_bed(Den_bed_path, Nean_bed_path, archaic_bed_path, archaic_bed_info_path):
    Den_bed_raw = pd.read_csv(Den_bed_path, sep="\t", header=None)
    Den_bed_raw[4] = 'src1'
    Nean_bed_raw = pd.read_csv(Nean_bed_path, sep="\t", header=None)
    Nean_bed_raw[4] = 'src2'
    df = pd.concat([Den_bed_raw, Nean_bed_raw])

    merged_intervals = []

    # 按 chr + 倒数两列 来分组
    for (chrom, col3, col4), group in df.groupby([0, 3, 4]):
        group = group.sort_values(by=1)  # start 排序
        current_start = None
        current_end = None

        for _, row in group.iterrows():
            start, end = row[1], row[2]

            if current_start is None:
                current_start, current_end = start, end
            else:
                # 合并重叠/相邻的区间
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    merged_intervals.append([chrom, current_start, current_end, col3, col4])
                    current_start, current_end = start, end

        merged_intervals.append([chrom, current_start, current_end, col3, col4])

    merged_bed = pd.DataFrame(merged_intervals, columns=[0, 1, 2, 3, 4])

    merged_bed = merged_bed.sort_values(by=[3, 1]).reset_index(drop=True)

    tmp = merged_bed[[0,1,2,3]].copy()
    tmp = tmp.drop_duplicates(keep = 'first')

    tmp.to_csv(archaic_bed_path, sep="\t", header=False, index=False)

    merged_bed.columns = ["Chr", "Start", "End", "Index","Archaic"]
    merged_bed['Archaic'].replace(1, "src1", inplace=True)
    merged_bed['Archaic'].replace(2, "src2", inplace=True)
    merged_bed['Length_kb'] = (merged_bed['End'] - merged_bed['Start']) / 1000
    merged_bed.to_csv(archaic_bed_info_path, sep="\t", header=False, index=False)

def get_best_matches(Archaic_infered_bed, Archaic_sim_info_bed, out_dir, prefix):
    infered_df = pd.read_csv(Archaic_infered_bed, sep="\t", header=None)
    infered_df = infered_df[[0, 1, 2, 3, 4, 6]]
    infered_df.columns = ["chr", "start", "end", "hap_id", "Archaic", "Score"]

    infered_df["sample_id"] = infered_df["hap_id"].apply(
        lambda x: x - 1 if x % 2 == 1 else x
    )

    infered_df["Infered_Length"] = (infered_df["end"] - infered_df["start"]) / 1000

    infered_df = infered_df.reset_index().rename(columns={"index": "infer_id"})


    # ----- 2. 读入模拟真值 -----
    sim_info_df = pd.read_csv(Archaic_sim_info_bed, sep="\t", header=None)
    sim_info_df = sim_info_df[[0, 1, 2, 3, 4, 5]]
    sim_info_df.columns = ["chr", "start", "end", "hap_id", "Archaic", "Sim_Length"]
    sim_info_df["sample_id"] = sim_info_df["hap_id"].apply(
    lambda x: x - 1 if x % 2 == 1 else x)
    sim_info_df["Sim_Length"] = (sim_info_df["end"] - sim_info_df["start"]) / 1000
    sim_info_df = sim_info_df.reset_index().rename(columns={"index": "sim_id"})



    # ----- 3. 合并 & 计算 overlap：infered -> sim -----
    inferdsim = infered_df.merge(
        sim_info_df,
        on="hap_id",
        how="left",
        suffixes=("_infered", "_sim")
    )

    start_max = np.maximum(inferdsim["start_infered"], inferdsim["start_sim"])
    end_min   = np.minimum(inferdsim["end_infered"],   inferdsim["end_sim"])

    inferdsim["overlap"] = (end_min - start_max).clip(lower=0) / 1000
    inferdsim["overlap_sim_ratio"]   = inferdsim["overlap"] / inferdsim["Sim_Length"]
    inferdsim["overlap_infer_ratio"] = inferdsim["overlap"] / inferdsim["Infered_Length"]

    # 丢掉没有匹配上的行 & 只保留 overlap 足够大的
    inferdsim_valid = inferdsim.dropna(subset=["overlap_sim_ratio"]).copy()
    # 这个阈值你可以自己调，比如 0.5 或 0.0
    # inferdsim_valid = inferdsim_valid[inferdsim_valid["overlap_sim_ratio"] > 0.8].copy()

    # 对每个推断片段 infer_id，选择 overlap_sim_ratio 最大的那一条
    best_idx_infersim = (
        inferdsim_valid
        .groupby("infer_id")["overlap_sim_ratio"]
        .idxmax()
        .dropna()
        .astype(int)
    )

    best_matches_infersim = inferdsim_valid.loc[best_idx_infersim].copy()

    best_matches_infersim = best_matches_infersim[
        [
            "chr_infered", "start_infered", "end_infered",
            "hap_id", "sample_id_infered","Infered_Length", "Score",
            "Archaic_infered", "Archaic_sim",
            "overlap_sim_ratio", "overlap_infer_ratio", "overlap",
            "chr_sim", "start_sim", "end_sim", "Sim_Length",
        ]
    ]
    best_matches_infersim.columns = [
        "chr", "start", "end",
        "hap_id", "sample_id","Infered_Length", "Score",
        "Archaic_infered", "Archaic_sim",
        "overlap_sim_ratio", "overlap_infer_ratio", "overlap",
        "chr_sim", "start_sim", "end_sim", "Sim_Length",
    ]

    # ----- 4. 再来一遍：sim -> infered （每个真值段找最好的推断段） -----
    siminfer = sim_info_df.merge(
        infered_df,
        on="hap_id",
        how="left",
        suffixes=("_sim", "_infered")
    )

    start_max2 = np.maximum(siminfer["start_sim"], siminfer["start_infered"])
    end_min2   = np.minimum(siminfer["end_sim"],   siminfer["end_infered"])

    siminfer["overlap"] = (end_min2 - start_max2).clip(lower=0) / 1000
    siminfer["overlap_sim_ratio"]   = siminfer["overlap"] / siminfer["Sim_Length"]
    siminfer["overlap_infer_ratio"] = siminfer["overlap"] / siminfer["Infered_Length"]

    siminfer_valid = siminfer.dropna(subset=["overlap_sim_ratio"]).copy()
    # siminfer_valid = siminfer_valid[siminfer_valid["overlap_sim_ratio"] > 0.8].copy()

    # 对每个真值片段 sim_id，选择 overlap_infer_ratio 最大的那一条
    best_idx_siminfer = (
        siminfer_valid
        .groupby("sim_id")["overlap_sim_ratio"]
        .idxmax()
        .dropna()
        .astype(int)
    )

    best_matches_siminfer = siminfer_valid.loc[best_idx_siminfer].copy()

    best_matches_siminfer = best_matches_siminfer[
        [
            "chr_sim", "start_sim", "end_sim",
            "hap_id", "sample_id_sim","Sim_Length",
            "Archaic_sim", "Archaic_infered", "Score",
            "overlap_sim_ratio", "overlap_infer_ratio", "overlap",
            "chr_infered", "start_infered", "end_infered", "Infered_Length",
        ]
    ]
    best_matches_siminfer.columns = [
        "chr", "start", "end",
        "hap_id", "sample_id","Sim_Length",
        "Archaic_sim", "Archaic_infered", "Score",
        "overlap_sim_ratio", "overlap_infer_ratio", "overlap",
        "chr_infered", "start_infered", "end_infered", "Infered_Length",
    ]

    # ----- 5. 写出两个方向的 bed 文件 -----
    out_path_infersim = out_dir / f"infersim.{prefix}.bed"
    out_path_siminfer = out_dir / f"siminfer.{prefix}.bed"

    best_matches_infersim.to_csv(out_path_infersim, sep="\t", index=False)
    best_matches_siminfer.to_csv(out_path_siminfer, sep="\t", index=False)

    # 返回两个 df
    return best_matches_infersim, best_matches_siminfer

def Filter_Score_Length(Archaic_infered_bed, out_dir, score_cutoff, length_cutoff, prefix):
    df = pd.read_csv(Archaic_infered_bed, sep="\t", header=None)
    df = df[df[6] >= score_cutoff]
    df = df[df[2] - df[1] >= length_cutoff * 1000]
    # prefix = f'PostFilter_score_{score_cutoff}_length_{length_cutoff}kb'
    prefix = f"{prefix}"

    df.to_csv(out_dir / f"src.{prefix}.bed", sep="\t", index=False, header=None)
    df[df[4] == 1].to_csv(out_dir / f"src1.{prefix}.bed", sep="\t", index=False, header=None)
    df[df[4] == 2].to_csv(out_dir / f"src2.{prefix}.bed", sep="\t", index=False, header=None)
    return prefix

def analyze_infer_sim_13(best_matches_infersim,best_matches_siminfer,min_overlap_sim_ratio=0,exchange=False):
    # ------------------------------------------------------------------
    # 1. 读取并处理
    # ------------------------------------------------------------------
    df = best_matches_infersim
    df2 = best_matches_siminfer[['Infered_Length', 'Score', 'overlap_sim_ratio','Archaic_infered', 'Archaic_sim']]

    # overlap=0 → non-archaic
    df.loc[df['overlap_sim_ratio'] <= min_overlap_sim_ratio, 'Archaic_sim'] = 0
    df2.loc[df2['overlap_sim_ratio'] <= min_overlap_sim_ratio, 'Archaic_infered'] = 0

    # 统一真实标签
    # df['Archaic_sim'] = df['Archaic_sim'].replace({'src1': 1, 'src2': 2}).astype(int)
    if exchange:
        df['Archaic_sim'] = df['Archaic_sim'].replace({'src1': 2, 'src2': 1}).astype(int)
        df2['Archaic_sim'] = df2['Archaic_sim'].replace({'src1': 2, 'src2': 1}).astype(int)
    else:
        df['Archaic_sim'] = df['Archaic_sim'].replace({'src1': 1, 'src2': 2}).astype(int)
        df2['Archaic_sim'] = df2['Archaic_sim'].replace({'src1': 1, 'src2': 2}).astype(int)

    # ------------------------------------------------------------------
    # 2. overall archaic ratio
    # ------------------------------------------------------------------
    Archaic_all_count = (df['Archaic_sim'] != 0).mean() * 100
    Archaic_all_recall = (df2['Archaic_infered'] != 0).mean() * 100

    # ------------------------------------------------------------------
    # 3. 每个推断类别中真实 archaic 比例（hit rate）
    # ------------------------------------------------------------------
    infer_hit_rate = (
        df.groupby('Archaic_infered')['Archaic_sim']
          .apply(lambda x: (x != 0).mean())
    )
    recall_hit_rate = (
        df2.groupby('Archaic_sim')['Archaic_infered']
          .apply(lambda x: (x != 0).mean())
    )
    print(recall_hit_rate)

    # 若某些 infered 类别不存在，补 0
    infer_hit_rate = infer_hit_rate.reindex([1,2,3], fill_value=0) * 100
    recall_hit_rate = recall_hit_rate.reindex([1,2], fill_value=0) * 100

    df = df[df['Archaic_sim'] != 0]
    df['Distance_End'] = df['end'] - df['end_sim']
    df['Distance_Start'] = df['start'] - df['start_sim']
    df['Abs_Distance'] = df['Distance_End'].abs() + df['Distance_Start'].abs()
    df['Relative_Distance'] = df['Abs_Distance'] / 1000 / df['Sim_Length']

    Dis_End_mean = df['Distance_End'].mean()
    Dis_Start_mean = df['Distance_Start'].mean()
    Dis_Abs_mean = df['Abs_Distance'].mean()
    Dis_Abs_std = df['Abs_Distance'].std()
    Dis_Rel_mean = df['Relative_Distance'].mean()
    Dis_Rel_std = df['Relative_Distance'].std()

    # ------------------------------------------------------------------
    # 4. 混淆矩阵（真实 0/1/2 × 预测 1/2/3）
    # ------------------------------------------------------------------
    y_true = df['Archaic_sim'].astype(int)
    y_pred = df['Archaic_infered'].astype(int)

    labels_pred = [1,2,3]   # columns

    cm = confusion_matrix(y_true, y_pred, labels=labels_pred)

    acc = (cm[0][0] + cm[1][1] + cm[2][2])/cm.sum() * 100
    DM = cm[0][2] / (cm[0][2] + cm[1][2] + cm[2][2]) * 100
    NM = cm[1][2] / (cm[0][2] + cm[1][2] + cm[2][2]) * 100
    DN = cm[0][1] / (cm[0][1] + cm[1][1] + cm[2][1]) * 100
    ND = cm[1][0] / (cm[1][0] + cm[2][0] + cm[0][0]) * 100

    # ------------------------------------------------------------------
    # 5. 返回 13 个值
    # ------------------------------------------------------------------
    return Archaic_all_count , infer_hit_rate.loc[1] , infer_hit_rate.loc[2] , infer_hit_rate.loc[3] , Archaic_all_recall , recall_hit_rate.loc[1] , recall_hit_rate.loc[2] , acc , DM , NM , DN , ND,Dis_End_mean,Dis_Start_mean,Dis_Abs_mean,Dis_Abs_std,Dis_Rel_mean,Dis_Rel_std


def main():
    args = parse_args()
    seed = args.seed
    demo = args.demo
    nref = args.nref
    ntgt = args.ntgt

    as3dir = Path("/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker3.0")
    simdir = Path("/home/linhuanyu/share1/20_AS3/results/simulated_data")
    out_dir = as3dir / demo / f"nref_{nref}" / f"ntgt_{ntgt}" / str(seed)
    simdata_dir = simdir / demo / f"nref_{nref}" / f"ntgt_{ntgt}" / str(seed)

    print(f"seed: {seed}")
    print(f"demo: {demo}")

    if demo in ["AS2_HumanNeanderthalDenisovan", "ChimpBonoboGhost","HumanArchaic","HumanNeanderthalDenisovan"]:
        src_sim_bed = simdata_dir / f"sim2src.src.introgressed.tracts.bed"
        src_sim_info_bed = simdata_dir / f"sim2src.src.introgressed.tracts.info.bed"
        src1_sim_bed     = simdata_dir / f"sim2src.src1.introgressed.tracts.bed"
        src2_sim_bed    = simdata_dir / f"sim2src.src2.introgressed.tracts.bed"
        # process_Archaic_bed(src1_sim_bed, src2_sim_bed, src_sim_bed, src_sim_info_bed)
    else:
        src_sim_bed = simdata_dir / f"sim1src.introgressed.tracts.bed"
        src_sim_info_bed = simdata_dir / f"sim1src.introgressed.tracts.info.bed"
        src1_sim_bed = simdata_dir / f"sim1src.introgressed.tracts.bed"
        src2_sim_bed = simdata_dir / f"sim1src.introgressed.tracts.bed"
        # process_Archaic_bed(src1_sim_bed, src2_sim_bed, src_sim_bed, src_sim_info_bed)

    AS3_premerge_bed = out_dir / f"introgression_prediction.raw.bk.bed"
    AS3_bed_for_merge = out_dir / f"introgression_prediction.raw.bed"
    # Filter(AS3_premerge_bed, AS3_bed_for_merge, min_length = 5, min_score = 0)
    AS3_postmerge_bed = out_dir / f"Raw.1211.bed"

    # cmd = (
    #     f"cd {out_dir} && "
    #     f"/home/linhuanyu/02_Software/ArchaicSeeker3_memFast_TwoStage/merge_bed_segments.py "
    #     f"-i {AS3_bed_for_merge} "
    #     f"-o {AS3_postmerge_bed} -d 10000"
    # )
    # subprocess.run(cmd, shell=True, check=True)

    # Filter_Score_Length(AS3_postmerge_bed, out_dir = out_dir, score_cutoff = 0.6, length_cutoff = 10 ,prefix = "1211")

    out_accuracy = out_dir / f"AS3_1211.sample.accuracy"
    src_AS3_bed = out_dir / f"src.1208.bed"
    src1_AS3_bed = out_dir / f"src1.1208.bed"
    src2_AS3_bed = out_dir / f"src2.1208.bed"

    Archaic_prec, Archaic_rec, Archaic_f1, Archaic_ratio = cal_accuracy_sample(src_sim_bed, src_AS3_bed)
    if demo in ["AS2_HumanNeanderthalDenisovan", "HumanNeanderthalDenisovan"]:
        Den_prec, Den_rec, Den_f1, Den_ratio                 = cal_accuracy_sample(src2_sim_bed, src1_AS3_bed)
        Nean_prec, Nean_rec, Nean_f1, Nean_ratio             = cal_accuracy_sample(src1_sim_bed, src2_AS3_bed)
    else:
    #     Archaic_prec, Archaic_rec, Archaic_f1, Archaic_ratio = cal_accuracy_hap(src_sim_bed, src_AS3_bed)
        Den_prec, Den_rec, Den_f1, Den_ratio                 = cal_accuracy_sample(src1_sim_bed, src1_AS3_bed)
        Nean_prec, Nean_rec, Nean_f1, Nean_ratio             = cal_accuracy_sample(src2_sim_bed, src2_AS3_bed)

    Archaic_ratio = Archaic_ratio * 100
    Den_ratio     = Den_ratio * 100
    Nean_ratio    = Nean_ratio * 100

    best_matches_infersim,best_matches_siminfer = get_best_matches(
        src_AS3_bed,
        src_sim_info_bed,
        out_dir,
        "1211")

    if demo in ["AS2_HumanNeanderthalDenisovan", "HumanNeanderthalDenisovan"]:
        pre_seg_all, pre_seg_1, pre_seg_2, pre_seg_3, recall_seg_all,recall_seg_1,recall_seg_2,acc, DM, NM, DN, ND ,dis_end_mean,dis_start_mean,dis_abs_mean,dis_abs_std,dis_rel_mean,dis_rel_std= analyze_infer_sim_13(best_matches_infersim,best_matches_siminfer, min_overlap_sim_ratio=0.8, exchange=True)
    else:
        pre_seg_all, pre_seg_1, pre_seg_2, pre_seg_3, recall_seg_all,recall_seg_1,recall_seg_2,acc, DM, NM, DN, ND ,dis_end_mean,dis_start_mean,dis_abs_mean,dis_abs_std,dis_rel_mean,dis_rel_std= analyze_infer_sim_13(best_matches_infersim, best_matches_siminfer, min_overlap_sim_ratio=0.8)

    with open(out_accuracy, "w") as f:
        f.write(
            f"1211\t{demo}\t{nref}\t{ntgt}\t{seed}\t"
            f"{Archaic_prec}\t{Archaic_rec}\t{Archaic_f1}\t{Archaic_ratio}\t"
            f"{Den_prec}\t{Den_rec}\t{Den_f1}\t{Den_ratio}\t"
            f"{Nean_prec}\t{Nean_rec}\t{Nean_f1}\t{Nean_ratio}\t"
            f"{pre_seg_all}\t{pre_seg_1}\t{pre_seg_2}\t{pre_seg_3}\t{recall_seg_all}\t{recall_seg_1}\t{recall_seg_2}\t{acc}\t{DM}\t{NM}\t{DN}\t{ND}\t{dis_end_mean}\t{dis_start_mean}\t{dis_abs_mean}\t{dis_abs_std}\t{dis_rel_mean}\t{dis_rel_std}\n"
        )


if __name__ == "__main__":
    main()



