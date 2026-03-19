import os
import numpy as np
import yaml
import subprocess

# === 载入配置 ===
with open("config/snakemake/config.yaml") as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]
demo_model_list = config["demo_models"]
nref_list = [10, 50]
ntgt_list = [10]
np.random.seed(config["seed"])
replicates = config["replicates"]
seed_list = list(np.random.randint(1, 2**31, replicates))

log_base = os.path.join(output_dir, "logs")
os.makedirs(log_base, exist_ok=True)

for demog in demo_model_list:
    for nref in nref_list:
        for ntgt in ntgt_list:
            for seed in seed_list:
                try:
                    prefix = f"inference/ArchaicSeeker3.1/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}"
                    full_prefix = os.path.join(output_dir, prefix)
                    os.makedirs(full_prefix, exist_ok=True)

                    tgt_vcf = os.path.join(full_prefix, "sim.tgt.vcf.gz")
                    intro_raw = os.path.join(full_prefix, "introgression_prediction.bed")
                    intro_bed = os.path.join(full_prefix, "AS3_Mamba_Smoother_4096_2048_5216.bed")
                    log_file = os.path.join(log_base, f"process_bed_{demog}_{nref}_{ntgt}_{seed}.log")

                    # Step 1: query sample list
                    tgt_sample_file = os.path.join(full_prefix, "tgt_samples.txt")
                    with open(log_file, 'a') as log:
                        subprocess.run(["bcftools", "query", "-l", tgt_vcf], stdout=open(tgt_sample_file, 'w'), stderr=log)

                    # Step 2: generate hapmap
                    hapmap_file = os.path.join(full_prefix, "hapmap0424.txt")
                    with open(tgt_sample_file) as f_in, open(hapmap_file, 'w') as f_out:
                        for i, line in enumerate(f_in):
                            sample = line.strip()
                            f_out.write(f"{2*i}\t{sample}_1\n")
                            f_out.write(f"{2*i+1}\t{sample}_2\n")

                    # Step 3: map intro_raw and mosaic_raw with hapmap
                    def map_with_hapmap(hapmap_path, input_bed, output_bed):
                        hapmap = {}
                        with open(hapmap_path) as f:
                            for line in f:
                                idx, name = line.strip().split()
                                hapmap[idx] = name
                        with open(input_bed) as f_in, open(output_bed, 'w') as f_out:
                            for line in f_in:
                                parts = line.strip().split()
                                parts[3] = hapmap.get(parts[3], parts[3])
                                f_out.write("\t".join(parts) + "\n")

                    map_with_hapmap(hapmap_file, intro_raw, intro_bed)

                    print(f"✅ 处理完成: {prefix}")

                except Exception as e:
                    print(f"❌ 错误处理: {prefix}", e)