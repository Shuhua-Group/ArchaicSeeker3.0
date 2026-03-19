import os
import numpy as np
from utils.utils import process_ibdmix_output, cal_accuracy
import yaml
import sys
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import process_archaicseeker3_1src_output, cal_accuracy

# === 读取配置文件 ===
with open("config/snakemake/config.yaml") as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]
replicates = config["replicates"]
np.random.seed(config["seed"])

seed_list = list(np.random.randint(1, 2**31, replicates))
demo_model_list = config["1src_demo_models"]
cutoff_list = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50]
nref_list = [10, 50]
ntgt_list = [1,10]

results = []

for demog in demo_model_list:
    for nref in nref_list:
        for ntgt in ntgt_list:
            for seed in seed_list:
                base_path = os.path.join(
                    output_dir,
                    f"inference/IBDmix/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}"
                )
                ibdmix_res = os.path.join(base_path, "ibdmix_output.txt")
                true_tracts = os.path.join(
                    output_dir,
                    f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim1src.introgressed.tracts.bed"
                )
                inferred_base = os.path.join(base_path, "IBDmix.out.cutoff")

                for cutoff in cutoff_list:
                    inferred_file = f"{inferred_base}.{cutoff}.bed"
                    acc_file = f"{base_path}/IBDmix.out.cutoff.{cutoff}.accuracy"

                    # 运行推断输出处理及准确率计算
                    process_ibdmix_output(ibdmix_res, inferred_file, float(cutoff))
                    precision, recall = cal_accuracy(true_tracts, inferred_file)

                    # 保存结果
                    with open(acc_file, 'w') as o:
                        o.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\t{cutoff}\t{precision:.2f}\t{recall:.2f}\n")

                    results.append((demog, f"nref_{nref}_ntgt_{ntgt}", cutoff, precision, recall))
                    print(f"✅ Finished: {demog} | nref={nref}, ntgt={ntgt}, seed={seed}, cutoff={cutoff}")

# === 汇总结果表 ===
summary_path = os.path.join(output_dir, "inference/IBDmix/IBDmix_1src_accuracy.txt")
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, 'w') as f:
    f.write("demography\tsample\tcutoff\tprecision\trecall\n")
    for row in results:
        f.write("{}\t{}\t{}\t{:.2f}\t{:.2f}\n".format(*row))

print("\n🎉 所有任务完成！结果汇总文件保存在:", summary_path)