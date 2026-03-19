import os
import numpy as np
import sys
import yaml

# 加入 utils 路径
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import process_archaicseeker3_output, process_archaicseeker3_1src_output, cal_accuracy

# === 载入配置 ===
with open("config/snakemake/config.yaml") as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]
replicates = config["replicates"]
demo_model_list = config["2src_demo_models"]
nref_list = [10, 50]
ntgt_list = [1, 10]
seed_list = list(np.random.RandomState(config["seed"]).randint(1, 2**31, replicates))
cutoff_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

log_file = os.path.join(output_dir, "logs", "manual_cutoff_2src_eval.log")
error_log = "/home/linhuanyu/AS3_cutoff_2src_errors.log"
summary_out = os.path.join(output_dir, "inference/ArchaicSeeker3.0/AS3_Transformer_Smoother_200_100_1000_4803_2src_accuracy.txt")

os.makedirs(os.path.dirname(summary_out), exist_ok=True)

with open(summary_out, 'w') as fout:
    fout.write("demography\tsample\tsrc\tcutoff\tprecision\trecall\n")

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

                            merged_true_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.introgressed.tracts.bed")

                            with open(merged_true_file, 'w') as fout_merge:
                                for source_file in [true_file_1, true_file_2]:
                                    with open(source_file) as f:
                                        for line in f:
                                            parts = line.strip().split()
                                            if len(parts) >= 3:
                                                fout_merge.write("\t".join(parts[:3]) + "\n")


                            bed_out_1 = os.path.join(prefix, f"AS3_Transformer_Smoother_200_100_1000_4803.out.cutoff.{cutoff}.src1.bed")
                            bed_out_2 = os.path.join(prefix, f"AS3_Transformer_Smoother_200_100_1000_4803.out.cutoff.{cutoff}.src2.bed")
                            bed_out = os.path.join(prefix, f"AS3_Transformer_Smoother_200_100_1000_4803.out.cutoff.{cutoff}.bed")
                            acc_out = os.path.join(prefix, f"AS3_Transformer_Smoother_200_100_1000_4803.out.cutoff.{cutoff}.accuracy")

                            if not os.path.exists(seg_file):
                                print(f"[SKIP] {seg_file} 不存在，跳过该任务。")
                                continue

                            os.makedirs(os.path.dirname(bed_out), exist_ok=True)

                            process_archaicseeker3_output(seg_file, bed_out_1, bed_out_2, cutoff)
                            process_archaicseeker3_1src_output(seg_file, bed_out, cutoff)
                            # precision_1, recall_1 = cal_accuracy(true_file_1, bed_out_1)
                            # precision_2, recall_2 = cal_accuracy(true_file_2, bed_out_2)
                            precision, recall = cal_accuracy(merged_true_file, bed_out)

                            # 组合1：true1 ↔ pred1, true2 ↔ pred2
                            p11, r11 = cal_accuracy(true_file_1, bed_out_1)
                            p22, r22 = cal_accuracy(true_file_2, bed_out_2)

                            # 组合2：true1 ↔ pred2, true2 ↔ pred1
                            p12, r12 = cal_accuracy(true_file_1, bed_out_2)
                            p21, r21 = cal_accuracy(true_file_2, bed_out_1)

                            # 比较 recall 总和
                            recall_sum_1 = r11 + r22
                            recall_sum_2 = r12 + r21

                            # 选择更优组合（基于 recall 总和）
                            if recall_sum_1 >= recall_sum_2:
                                precision_1, recall_1 = p11, r11  # ↔ true_file_1
                                precision_2, recall_2 = p22, r22  # ↔ true_file_2
                                match_note = "✅ true1 ↔ pred1, true2 ↔ pred2 (by recall)"
                            else:
                                precision_1, recall_1 = p12, r12  # ↔ true_file_1
                                precision_2, recall_2 = p21, r21  # ↔ true_file_2
                                match_note = "✅ true1 ↔ pred2, true2 ↔ pred1 (by recall)"

                            print(match_note)

                            # 写入主输出文件
                            fout.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc1\t{cutoff}\t{precision_1:.2f}\t{recall_1:.2f}\n")
                            fout.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc2\t{cutoff}\t{precision_2:.2f}\t{recall_2:.2f}\n")
                            fout.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc\t{cutoff}\t{precision:.2f}\t{recall:.2f}\n")

                            # 写入精简 acc 文件
                            with open(acc_out, 'w') as acc_f:
                                acc_f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc1\t{cutoff}\t{precision_1:.2f}\t{recall_1:.2f}\n")
                                acc_f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc2\t{cutoff}\t{precision_2:.2f}\t{recall_2:.2f}\n")
                                acc_f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc\t{cutoff}\t{precision:.2f}\t{recall:.2f}\n")

                            print(f"✅ 成功处理: {demog}, nref={nref}, ntgt={ntgt}, seed={seed}, cutoff={cutoff}")

                        except Exception as e:
                            with open(error_log, 'a') as ferr:
                                ferr.write(f"\n❌ Error: {demog}, nref={nref}, ntgt={ntgt}, seed={seed}, cutoff={cutoff}\n")
                                ferr.write(str(e) + "\n")
                            print(f"[ERROR] {demog}, nref={nref}, ntgt={ntgt}, seed={seed}, cutoff={cutoff}", e)