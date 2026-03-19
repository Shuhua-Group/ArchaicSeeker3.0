import os
import numpy as np
import sys
import yaml

# 加入 utils 路径
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import process_archaicseeker3_1src_output, cal_accuracy

# === 载入配置 ===
with open("config/snakemake/config.yaml") as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]
replicates = config["replicates"]
demo_model_list = config["2src_demo_models"]
nref_list = [10, 50]
ntgt_list = [1]
seed_list = list(np.random.RandomState(config["seed"]).randint(1, 2**31, replicates))
cutoff_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

log_file = os.path.join(output_dir, "logs", "manual_cutoff_2src_eval.log")
error_log = "/home/linhuanyu/AS3_cutoff_2src_errors.log"
summary_out = os.path.join(output_dir, "inference/ArchaicSeeker3.0/AS3_Transformer_Smoother_200_100_1000_4803_2src_accuracy.merged.txt")

os.makedirs(os.path.dirname(summary_out), exist_ok=True)

with open(summary_out, 'w') as fout:
    fout.write("demography\tsample\tcutoff\tprecision\trecall\n")

    for demog in demo_model_list:
        for nref in nref_list:
            for ntgt in ntgt_list:
                for seed in seed_list:
                    for cutoff in cutoff_list:
                        try:
                            prefix = os.path.join(output_dir, f"inference/ArchaicSeeker3.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")
                            seg_file = os.path.join(prefix, "AS3_Transformer_Smoother_200_100_1000_4803.bed")
                            true_file_1 = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src1.introgressed.tracts.bed")
                            true_file_2 = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src2.introgressed.tracts.bed")

                            merged_true_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.merged.introgressed.tracts.bed")

                            # 拼接 src1 和 src2 成 merged 文件
                            with open(merged_true_file, 'w') as fout_merge:
                                with open(true_file_1) as f1:
                                    for line in f1:
                                        fout_merge.write(line)
                                with open(true_file_2) as f2:
                                    for line in f2:
                                        fout_merge.write(line)

                            bed_out = os.path.join(prefix, f"AS3_Transformer_Smoother_200_100_1000_4803.out.cutoff.{cutoff}.merged.bed")
                            acc_out = os.path.join(prefix, f"AS3_Transformer_Smoother_200_100_1000_4803.out.cutoff.{cutoff}.merged.accuracy")

                            if not os.path.exists(seg_file):
                                print(f"[SKIP] {seg_file} 不存在，跳过该任务。")
                                continue

                            os.makedirs(os.path.dirname(bed_out), exist_ok=True)
                            process_archaicseeker3_1src_output(seg_file, bed_out, cutoff)
                            precision, recall = cal_accuracy(merged_true_file, bed_out)

                            # 写入主汇总文件
                            fout.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\t{cutoff}\t{precision}\t{recall}\n")

                            # 写入每个任务自己的 .accuracy 文件
                            with open(acc_out, 'w') as acc_f:
                                acc_f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\t{cutoff}\t{precision}\t{recall}\n")

                            print(f"✅ 成功处理: {demog}, nref={nref}, ntgt={ntgt}, seed={seed}, cutoff={cutoff}")

                        except Exception as e:
                            with open(error_log, 'a') as ferr:
                                ferr.write(f"\n❌ Error: {demog}, nref={nref}, ntgt={ntgt}, seed={seed}, cutoff={cutoff}\n")
                                ferr.write(str(e) + "\n")
                            print(f"[ERROR] {demog}, nref={nref}, ntgt={ntgt}, seed={seed}, cutoff={cutoff}", e)