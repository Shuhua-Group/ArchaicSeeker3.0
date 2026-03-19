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
from acc import cal_accuracy_hap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Task for ArchaicSeeker3")
    parser.add_argument("--seed", type=str, required=True, help="seed")
    parser.add_argument("--prefix", type=str, required=True, help="prefix")
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


def get_best_matches(Archaic_infered_bed, Archaic_sim_info_bed, out_dir, prefix):

    infered_df = pd.read_csv(Archaic_infered_bed, sep="\t", header=None)
    infered_df = infered_df[[0,1,2,3,4,6]]
    infered_df.columns = ["chr", "start", "end", "hap_id", "Archaic", "Score"]
    infered_df['Infered_Length'] = (infered_df['end'] - infered_df['start'])/1000
    infered_df = infered_df.reset_index().rename(columns={'index': 'infer_id'})

    sim_info_df = pd.read_csv(Archaic_sim_info_bed, sep="\t", header=None)
    sim_info_df = sim_info_df[[0,1,2,3,4,5]]
    sim_info_df.columns = ["chr", "start", "end", "hap_id", "Archaic", "Sim_Length"]

    inferdsim = infered_df.merge(sim_info_df, on="hap_id", how="left",suffixes=("_infered", "_sim"))
    start_max = np.maximum(inferdsim["start_infered"], inferdsim["start_sim"])
    end_min   = np.minimum(inferdsim["end_infered"],   inferdsim["end_sim"])
    inferdsim["overlap"] = (end_min - start_max).clip(lower=0)/1000
    inferdsim["overlap_sim_ratio"] = inferdsim["overlap"] / inferdsim["Sim_Length"]
    inferdsim["overlap_infer_ratio"] = inferdsim["overlap"] / inferdsim["Infered_Length"]

    best_idx_infersim = inferdsim.groupby("infer_id")["overlap_sim_ratio"].idxmax()
    best_matches_infersim = inferdsim.loc[best_idx_infersim].copy()
    best_matches_infersim = best_matches_infersim[["chr_infered", "start_infered", "end_infered", "hap_id", "Infered_Length","Score","Archaic_infered", "Archaic_sim","overlap_sim_ratio","overlap_infer_ratio","overlap","start_sim","end_sim","Sim_Length",]]
    best_matches_infersim.columns = ["chr", "start", "end", "hap_id","Infered_Length","Score","Archaic_infered", "Archaic_sim","overlap_sim_ratio","overlap_infer_ratio","overlap","start_sim","end_sim","Sim_Length"]
    best_matches_infersim.to_csv(out_dir / f"infersim.{prefix}.bed", sep="\t", index=False)
    return best_matches_infersim

def Filter_Score_Length(Archaic_infered_bed, out_dir, score_cutoff, length_cutoff, prefix):
    df = pd.read_csv(Archaic_infered_bed, sep="\t", header=None)
    df = df[df[6] >= score_cutoff]
    df = df[df[2] - df[1] >= length_cutoff * 1000]
    # prefix = f'PostFilter_score_{score_cutoff}_length_{length_cutoff}kb'
    prefix = f"{prefix}"

    df.to_csv(out_dir / f"Archaic.{prefix}.bed", sep="\t", index=False, header=None)
    df[df[4] == 1].to_csv(out_dir / f"Den.{prefix}.bed", sep="\t", index=False, header=None)
    df[df[4] == 2].to_csv(out_dir / f"Nean.{prefix}.bed", sep="\t", index=False, header=None)
    return prefix

def analyze_infer_sim_13(df,min_overlap_sim_ratio=0):
    # ------------------------------------------------------------------
    # 1. 读取并处理
    # ------------------------------------------------------------------
    df = df[['Infered_Length', 'Score', 'overlap_sim_ratio','Archaic_infered', 'Archaic_sim']]

    # overlap=0 → non-archaic
    df.loc[df['overlap_sim_ratio'] <= min_overlap_sim_ratio, 'Archaic_sim'] = 0

    # 统一真实标签
    df['Archaic_sim'] = df['Archaic_sim'].replace({'Den': 1, 'Nean': 2}).astype(int)

    # ------------------------------------------------------------------
    # 2. overall archaic ratio
    # ------------------------------------------------------------------
    Archaic_all_count = (df['Archaic_sim'] != 0).mean() * 100

    # ------------------------------------------------------------------
    # 3. 每个推断类别中真实 archaic 比例（hit rate）
    # ------------------------------------------------------------------
    infer_hit_rate = (
        df.groupby('Archaic_infered')['Archaic_sim']
          .apply(lambda x: (x != 0).mean())
    )

    # 若某些 infered 类别不存在，补 0
    infer_hit_rate = infer_hit_rate.reindex([1,2,3], fill_value=0) * 100

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
    return Archaic_all_count , infer_hit_rate.loc[1] , infer_hit_rate.loc[2] , infer_hit_rate.loc[3] , acc , DM , NM , DN , ND


def main():
    args = parse_args()
    seed = args.seed
    prefix = args.prefix

    basedir = Path("/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/6_TgtFreq/")
    out_dir = basedir / str(seed) 
    res_dir = basedir / str(seed) / str(prefix)

    print(f"seed: {seed}")
    print(f"prefix: {prefix}")

    Archaic_sim_bed = out_dir / f"Archaic.sim.bed"
    Archaic_sim_info_bed = out_dir / f"Archaic.sim.info.bed"
    Den_sim_bed     = out_dir / f"Den.sim.bed"
    Nean_sim_bed    = out_dir / f"Nean.sim.bed"

    AS3_raw_bed = res_dir / "introgression_prediction.raw.bk.bed"
    # AS3_raw_snp = out_dir / "introgression_prediction.raw.snp.bk.gz"
    # AS3_raw_prob = out_dir / "introgression_prob_matrix.bk.txt"

    AS3_filtered_bed = res_dir / f"introgression_prediction.raw.bed"
    # AS3_filtered_snp = out_dir / f"introgression_prediction.raw.snp.gz"
    # AS3_filtered_prob = out_dir / f"introgression_prob_matrix.txt"

    Filter(AS3_raw_bed, AS3_filtered_bed, min_length = 5, min_score = 0)

    Archaic_infered_raw_bed = res_dir / f"Raw.{prefix}.bed"
    Archaic_infered_bed = res_dir / f"Archaic.{prefix}.bed"
    Den_infered_bed     = res_dir / f"Den.{prefix}.bed"
    Nean_infered_bed    = res_dir / f"Nean.{prefix}.bed"

    out_accuracy = out_dir / f"{prefix}.accuracy"

    # merge segments
    cmd = (
        f"cd {res_dir} && "
        f"/home/linhuanyu/02_Software/ArchaicSeeker3_memFast_TwoStage/merge_bed_segments.py "
        f"-i {AS3_filtered_bed} "
        f"-o {Archaic_infered_raw_bed} -d 10000"
    )
    subprocess.run(cmd, shell=True, check=True)

    Filter_Score_Length(Archaic_infered_raw_bed, out_dir = res_dir, score_cutoff = 0.6, length_cutoff = 10 ,prefix = prefix)

    # accuracy
    Archaic_prec, Archaic_rec, Archaic_f1, Archaic_ratio = cal_accuracy_hap(Archaic_sim_bed, Archaic_infered_bed)
    Den_prec, Den_rec, Den_f1, Den_ratio                 = cal_accuracy_hap(Den_sim_bed,     Den_infered_bed)
    Nean_prec, Nean_rec, Nean_f1, Nean_ratio             = cal_accuracy_hap(Nean_sim_bed,    Nean_infered_bed)

    Archaic_ratio = Archaic_ratio * 100
    Den_ratio     = Den_ratio * 100
    Nean_ratio    = Nean_ratio * 100

    best_matches_infersim = get_best_matches(
        Archaic_infered_bed,
        Archaic_sim_info_bed,
        res_dir,
        prefix)

    pre_seg_all, pre_seg_1, pre_seg_2, pre_seg_3, acc, DM, NM, DN, ND = analyze_infer_sim_13(best_matches_infersim, min_overlap_sim_ratio=0)

    with open(out_accuracy, "w") as f:
        f.write(
            f"{prefix}\t{seed}\t"
            f"{Archaic_prec}\t{Archaic_rec}\t{Archaic_f1}\t{Archaic_ratio}\t"
            f"{Den_prec}\t{Den_rec}\t{Den_f1}\t{Den_ratio}\t"
            f"{Nean_prec}\t{Nean_rec}\t{Nean_f1}\t{Nean_ratio}\t"
            f"{pre_seg_all}\t{pre_seg_1}\t{pre_seg_2}\t{pre_seg_3}\t{acc}\t{DM}\t{NM}\t{DN}\t{ND}\n"
        )


if __name__ == "__main__":
    main()



