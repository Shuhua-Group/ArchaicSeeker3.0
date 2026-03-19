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

from acc import cal_accuracy_hap

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Task for ArchaicSeeker3")
    parser.add_argument("--seed", type=str, required=True, help="seed")
    parser.add_argument("--prefix", type=str, required=True, help="prefix")
    return parser.parse_args()

def get_best_matches(infered_bed, sim_info_bed, out_dir, prefix):

    infered_df = pd.read_csv(infered_bed, sep="\t", header=None)
    infered_df = infered_df[[0,1,2,3,4,6]]
    infered_df.columns = ["chr", "start", "end", "hap_id", "Archaic", "Score"]
    infered_df['Infered_Length'] = (infered_df['end'] - infered_df['start'])/1000
    infered_df = infered_df.reset_index().rename(columns={'index': 'infer_id'})

    sim_info_df = pd.read_csv(sim_info_bed, sep="\t", header=None)
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

    sim_info_df = sim_info_df.reset_index().rename(columns={'index': 'id'})
    siminfered = sim_info_df.merge(infered_df, on="hap_id", how="left",suffixes=("_sim", "_infered"))
    start_max = np.maximum(siminfered["start_sim"], siminfered["start_infered"])
    end_min   = np.minimum(siminfered["end_sim"],   siminfered["end_infered"])
    siminfered['Infered_Length'] = (siminfered['end_infered'] - siminfered["start_infered"])/1000
    siminfered["overlap"] = (end_min - start_max).clip(lower=0)/1000
    siminfered["overlap_sim_ratio"] = siminfered["overlap"] / siminfered["Sim_Length"]
    siminfered["overlap_infer_ratio"] = siminfered["overlap"] / siminfered["Infered_Length"]

    best_idx_siminfered = siminfered.groupby("id")["overlap_sim_ratio"].idxmax()
    best_matches_siminfered = siminfered.loc[best_idx_siminfered].copy()
    best_matches_siminfered = best_matches_siminfered[["chr_sim", "start_sim", "end_sim","hap_id","Sim_Length","Archaic_sim", "Archaic_infered",  "overlap_sim_ratio","overlap_infer_ratio","overlap","start_infered","end_infered","Infered_Length","Score"]]
    best_matches_siminfered.columns = ["chr", "start", "end", "hap_id", "Sim_Length","Archaic_sim", "Archaic_infered", "overlap_sim_ratio","overlap_infer_ratio","overlap","start_infered","end_infered","Infered_Length","Score"]
    best_matches_siminfered.to_csv(out_dir / f"siminfered.{prefix}.bed", sep="\t", index=False)

def main():
    args = parse_args()
    seed = args.seed
    prefix = args.prefix

    
    basedir = Path("/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut/")
    out_dir = basedir / str(seed)

    print(f"seed: {seed}")
    print(f"prefix: {prefix}")

    Archaic_sim_info_bed = out_dir / f"Archaic.sim.info.bed"
    Archaic_infered_bed_score = out_dir / f"AS3_Merge_0.bed"

    get_best_matches(Archaic_infered_bed_score, Archaic_sim_info_bed, out_dir, prefix)

    print('Done!')

if __name__ == "__main__":
    main()




