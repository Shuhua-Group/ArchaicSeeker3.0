import pandas as pd
import numpy as np
import argparse
import os
import subprocess
from pathlib import Path
import sys, os
ACC_PATH = "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/acc.py"
ACC_DIR = os.path.dirname(ACC_PATH)
sys.path.append(ACC_DIR)

from acc import cal_accuracy_hap, cal_accuracy_sample
from sklearn.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Task for ArchaicSeeker3")
    parser.add_argument("--seed", type=str, required=True, help="seed")
    parser.add_argument("--prefix", type=str, required=True, help="prefix")
    return parser.parse_args()

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

def Filter_Score_Length(Archaic_infered_bed, out_dir, score_cutoff, length_cutoff):
    df = pd.read_csv(Archaic_infered_bed, sep="\t", header=None)
    df = df[df[6] >= score_cutoff]
    df = df[df[2] - df[1] >= length_cutoff * 1000]
    prefix = f'PostFilter_score_{score_cutoff}_length_{length_cutoff}kb'

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
    Archaic_all_count = (df['Archaic_sim'] != 0).mean()

    # ------------------------------------------------------------------
    # 3. 每个推断类别中真实 archaic 比例（hit rate）
    # ------------------------------------------------------------------
    infer_hit_rate = (
        df.groupby('Archaic_infered')['Archaic_sim']
          .apply(lambda x: (x != 0).mean())
    )

    # 若某些 infered 类别不存在，补 0
    infer_hit_rate = infer_hit_rate.reindex([1,2,3], fill_value=0)

    # ------------------------------------------------------------------
    # 4. 混淆矩阵（真实 0/1/2 × 预测 1/2/3）
    # ------------------------------------------------------------------
    y_true = df['Archaic_sim'].astype(int)
    y_pred = df['Archaic_infered'].astype(int)

    labels_pred = [1,2,3]   # columns
    labels_true = [0,1,2]   # rows

    cm = confusion_matrix(y_true, y_pred, labels=labels_pred)

    # cm[i][j] = true = labels_true[i]? pred = labels_pred[j]

    # 展开成 9 个值
    cm_dict = {
        "T0_P1": cm[0][0], "T0_P2": cm[0][1], "T0_P3": cm[0][2],
        "T1_P1": cm[1][0], "T1_P2": cm[1][1], "T1_P3": cm[1][2],
        "T2_P1": cm[2][0], "T2_P2": cm[2][1], "T2_P3": cm[2][2],
    }

    acc = (cm[0][0] + cm[1][1] + cm[2][2])/cm.sum()

    # ------------------------------------------------------------------
    # 5. 返回 13 个值
    # ------------------------------------------------------------------
    return Archaic_all_count, infer_hit_rate.loc[1], infer_hit_rate.loc[2], infer_hit_rate.loc[3], cm[0][0], cm[0][1], cm[0][2], cm[1][0], cm[1][1], cm[1][2], cm[2][0], cm[2][1], cm[2][2], acc

def main():
    args = parse_args()
    seed = args.seed
    prefix = args.prefix
    
    basedir = Path("/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut/")
    out_dir = basedir / str(seed)

    print(f"seed: {seed}")
    print(f"prefix: {prefix}")

    Archaic_sim_bed = out_dir / f"Archaic.sim.bed"
    Archaic_sim_info_bed = out_dir / f"Archaic.sim.info.bed"
    Den_sim_bed = out_dir / f"Den.sim.bed"
    Nean_sim_bed = out_dir / f"Nean.sim.bed"
    Archaic_infered_bed = out_dir / f"{prefix}.bed"
    Den_infered_bed = out_dir / f"Den.{prefix}.bed"
    Nean_infered_bed = out_dir / f"Nean.{prefix}.bed"

    out_accuracy = out_dir / f"{prefix}.accuracy"

    # 建议：如果想每次脚本运行都重写文件，用 "w"；
    # 如果想在同一个文件里追加不同 seed 的结果，可以改成 "a"。
    with open(out_accuracy, "w") as f:
        for score_cutoff in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for length_cutoff in [0, 1, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 40, 50]:

                # 1) 根据 score & length 过滤
                tmp_prefix = Filter_Score_Length(Archaic_infered_bed, out_dir,
                                                score_cutoff, length_cutoff)
                Archaic_bed = out_dir / f"Archaic.{tmp_prefix}.bed"
                Den_bed     = out_dir / f"Den.{tmp_prefix}.bed"
                Nean_bed    = out_dir / f"Nean.{tmp_prefix}.bed"

                # 如果这一组阈值太严格，生成的 Archaic_bed 为空，可以在这里做个防御性判断
                if not Archaic_bed.exists():
                    # 也可以选择写一行占位，视你需求而定
                    continue

                # 2) 计算 Den/Nean/总 Archaic 的 hap-level accuracy
                Archaic_prec, Archaic_rec, Archaic_f1, Archaic_ratio = cal_accuracy_hap(
                    Archaic_sim_bed, Archaic_bed
                )
                Den_prec, Den_rec, Den_f1, Den_ratio = cal_accuracy_hap(
                    Den_sim_bed, Den_bed
                )
                Nean_prec, Nean_rec, Nean_f1, Nean_ratio = cal_accuracy_hap(
                    Nean_sim_bed, Nean_bed
                )

                # 3) infer vs sim 的 segment-level 匹配
                # 这里建议用 tmp_prefix 写 infersim 文件，这样不同阈值不会互相覆盖
                best_matches_infersim = get_best_matches(
                    Archaic_bed,
                    Archaic_sim_info_bed,
                    out_dir,
                    tmp_prefix,   # <<< 这里用 tmp_prefix，而不是 prefix
                )

                # 如果 best_matches_infersim 为空，也可以跳过
                if best_matches_infersim.empty:
                    continue

                FP_seg_all, FP_seg_1, FP_seg_2, FP_seg_3, \
                T0_P1, T0_P2, T0_P3, \
                T1_P1, T1_P2, T1_P3, \
                T2_P1, T2_P2, T2_P3, acc = analyze_infer_sim_13(
                    best_matches_infersim,
                    min_overlap_sim_ratio=0
                )

                # 4) 写入一行结果到 accuracy 文件
                f.write(
                    f"{tmp_prefix}\t{seed}\t"
                    f"{Archaic_prec}\t{Archaic_rec}\t{Archaic_f1}\t{Archaic_ratio}\t"
                    f"{Den_prec}\t{Den_rec}\t{Den_f1}\t{Den_ratio}\t"
                    f"{Nean_prec}\t{Nean_rec}\t{Nean_f1}\t{Nean_ratio}\t"
                    f"{FP_seg_all}\t{FP_seg_1}\t{FP_seg_2}\t{FP_seg_3}\t{acc}\t"
                    f"{T0_P1}\t{T0_P2}\t{T0_P3}\t"
                    f"{T1_P1}\t{T1_P2}\t{T1_P3}\t"
                    f"{T2_P1}\t{T2_P2}\t{T2_P3}\n"
                )


if __name__ == "__main__":
    main()




