import os
import numpy as np
import yaml

# === 载入配置 ===
with open("/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/snakemake/config.yaml") as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]
replicates = config["replicates"]
demo_model_list = config["1src_demo_models"]
nref_list = [10, 50]
ntgt_list = [1,10]
seed_list = list(np.random.RandomState(config["seed"]).randint(1, 2**31, replicates))
# cutoff_list = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50]

with open("task_1src_list.txt", "w") as f:
    for demog in demo_model_list:
        for nref in nref_list:
            for ntgt in ntgt_list:
                for seed in seed_list:
                    # for cutoff in cutoff_list:
                    f.write(f"{demog} {nref} {ntgt} {seed}\n")