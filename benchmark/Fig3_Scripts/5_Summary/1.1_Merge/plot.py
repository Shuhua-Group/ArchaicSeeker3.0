import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon


# =====================================================================
# 读取 accuracy，画箱线图，并返回汇总所需的 DataFrame
# =====================================================================

def load_accuracy_df(merge_nums, accuracy_dir):
    dfs = []
    for merge_num in merge_nums:
        path = f"{accuracy_dir}/AS3_Merge_{merge_num}.accuracy"
        df_tmp = pd.read_csv(path, sep="\t", header=None)
        dfs.append(df_tmp)

    df = pd.concat(dfs, ignore_index=True)

    df.columns = [
        "prefix", "seed",
        "Archaic Precision", "Archaic Recall", "Archaic F1", "Archaic Ratio",
        "Denisovan Precision", "Denisovan Recall", "Denisovan F1", "Denisovan Ratio",
        "Neandertal Precision", "Neandertal Recall", "Neandertal F1", "Neandertal Ratio"
    ]

    df["Merge_num"] = df["prefix"].str.split("_").str[2]
    df.drop(columns=["prefix", "seed"], inplace=True)

    metrics = [
        "Archaic Precision", "Archaic Recall", "Archaic F1", "Archaic Ratio",
        "Denisovan Precision", "Denisovan Recall", "Denisovan F1", "Denisovan Ratio",
        "Neandertal Precision", "Neandertal Recall", "Neandertal F1", "Neandertal Ratio"
    ]

    return df, metrics


def plot_accuracy_boxplots_from_df(df, metrics, out_dir="."):
    for m in metrics:
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=df, x="Merge_num", y=m,
            color="#4C72B0"   # 单色深蓝
        )
        plt.xlabel("Merge Distance", fontsize=16)
        plt.ylabel(m, fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        save_path = f"{out_dir}/{m.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print("Saved boxplot:", save_path)


# =====================================================================
# 画 Segment Length 直方图（返回 KS 和 JS）
# =====================================================================

def plot_segment_length_hist(sim_path, infer_path, save_path, merge_num, bins=100):

    # ------- 真值 -------
    Sim_bed = pd.read_csv(sim_path, sep="\t", header=None)
    Sim_bed.columns = ["Chromosome", "Start", "End", "Hap"]
    Sim_bed["Length"] = (Sim_bed["End"] - Sim_bed["Start"]) / 1000
    Sim_bed = Sim_bed[(Sim_bed["Length"] >= 5) & (Sim_bed["Length"] <= 300)]

    # ------- 推断 -------
    Infer_bed = pd.read_csv(infer_path, sep="\t", header=None)
    Infer_bed = Infer_bed[[0, 1, 2]]
    Infer_bed.columns = ["Chromosome", "Start", "End"]
    Infer_bed["Length"] = (Infer_bed["End"] - Infer_bed["Start"]) / 1000
    Infer_bed = Infer_bed[(Infer_bed["Length"] >= 5) & (Infer_bed["Length"] <= 300)]

    if Sim_bed.empty or Infer_bed.empty:
        print("Empty after filtering, skip:", save_path)
        return np.nan, np.nan

    # KS
    ks_stat = ks_2samp(Sim_bed["Length"], Infer_bed["Length"]).statistic

    # JS
    hist_sim, bin_edges = np.histogram(Sim_bed["Length"], bins=bins, range=(5, 300))
    hist_inf, _ = np.histogram(Infer_bed["Length"], bins=bin_edges)

    p = hist_sim / hist_sim.sum() if hist_sim.sum() > 0 else np.zeros_like(hist_sim)
    q = hist_inf / hist_inf.sum() if hist_inf.sum() > 0 else np.zeros_like(hist_inf)

    js_dist = jensenshannon(p, q)
    
    # ------- 画图 -------
    plt.figure(figsize=(6, 6))

    plt.hist(
        Sim_bed["Length"], bins=bins, range=(5, 300),
        density=True, alpha=0.6, color="gray"
    )
    plt.hist(
        Infer_bed["Length"], bins=bins, range=(5, 300),
        density=True, alpha=0.6, color="#4C72B0"
    )

    plt.xlabel("Segment Length (kb)", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(5, 200)

    text_str = f"Merge Distance = {merge_num} \nKS = {ks_stat:.3f}\nJS = {js_dist:.3f}"

    plt.text(
        0.97, 0.97, text_str,
        ha="right", va="top",
        transform=plt.gca().transAxes,
        fontsize=16,
        # bbox=dict(facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)
    return ks_stat, js_dist


# =====================================================================
# 主入口
# =====================================================================

def run_all_plots(merge_nums, base_dir, out_dir="."):

    os.makedirs(out_dir, exist_ok=True)

    print("=== Loading Accuracy ===")
    acc_df, metrics = load_accuracy_df(merge_nums, base_dir)

    print("=== Plotting Accuracy Boxplots ===")
    plot_accuracy_boxplots_from_df(acc_df, metrics, out_dir)

    # 汇总 accuracy
    acc_summary = acc_df.groupby("Merge_num")[metrics].mean().T
    acc_summary = acc_summary[[str(m) for m in merge_nums if str(m) in acc_summary.columns]]

    # 加 KS 和 JS
    acc_summary.loc["KS"] = np.nan
    acc_summary.loc["JS"] = np.nan

    print("=== Plotting Length Distributions ===")

    sim_path = f"{base_dir}/Archaic.sim.bed"

    for merge_num in merge_nums:
        infer_path = f"{base_dir}/AS3_Merge_{merge_num}.bed"
        save_path = f"{out_dir}/AS3_Merge_{merge_num}_Length.png"

        ks_stat, js_dist = plot_segment_length_hist(
            sim_path, infer_path, save_path, merge_num, bins=100
        )

        col = str(merge_num)
        if col in acc_summary.columns:
            acc_summary.loc["KS", col] = ks_stat
            acc_summary.loc["JS", col] = js_dist

    summary_path = f"{out_dir}/Metrics_summary.csv"
    acc_summary.to_csv(summary_path)

    print("=== Saved summary to", summary_path, "===")



# =====================================================================
# 运行
# =====================================================================

run_all_plots(
    merge_nums=[0, 2500, 5000, 7500, 10000, 12500, 15000, 17500,20000],
    base_dir="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/Merge",
    out_dir="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/Merge/Figures"
)
