import sys
import os
import subprocess
import json

sys.path.insert(0, "./")

# ================= 工具函数 =================
def run_cmd(cmd):
    print(f"\n🚀 Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 动态提取样本名
def extract_sample_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "outgroup" in data:
        return ",".join(data["outgroup"])
    else:
        raise ValueError("JSON内容格式错误，应为包含'outgroup'字段的dict！")

# ================= 主要逻辑 =================

# 读取命令行参数
demog, nref, ntgt, seed = sys.argv[1:5]

# 配置
output_dir = "/home/linhuanyu/share1/20_AS3/results"
sim_vcf = f"{output_dir}/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.biallelic.vcf.gz"
out_dir = f"{output_dir}/inference/SkovHMM/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}"
os.makedirs(out_dir, exist_ok=True)

# 样本名列表文件
outgroup_json = f"config/SkovHMM/{demog}/individuals.ref{nref}.tgt1.json"

# 获取样本名字符串
sample_names = extract_sample_from_json(outgroup_json)

# 文件路径
group_freq = f"{out_dir}/sim.outgroup.freq"
mutrates_file = f"{out_dir}/sim.mutrates.bed"
obs_file_prefix = f"{out_dir}/obs.tsk_{nref}"
train_file = f"{out_dir}/trained.json"
merged_vcf = f"{output_dir}/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src.biallelic.vcf.gz"
decode_output = f"{out_dir}/decode.diploid.txt"

init_guess = f"config/SkovHMM/{demog}/Initialguesses.json"
src1_name = f"config/SkovHMM/{demog}/src1.ref{nref}.name.txt"
src2_name = f"config/SkovHMM/{demog}/src2.ref{nref}.name.txt"

# === 执行每一步 ===

# 1. 创建outgroup频率表
run_cmd(f"hmmix create_outgroup -ind={sample_names} -vcf={sim_vcf} -out={group_freq}")

# 2. 估计突变率
run_cmd(f"hmmix mutation_rate -outgroup={group_freq} -out={mutrates_file}")

# 3. 创建ingroup观测数据
run_cmd(f"hmmix create_ingroup -ind={sample_names} -vcf={sim_vcf} -out={obs_file_prefix} -outgroup={group_freq}")

# 4. 训练模型
first_sample = sample_names.split(",")[0]
run_cmd(f"hmmix train -obs={obs_file_prefix}.{first_sample}.txt -mutrates={mutrates_file} -param={init_guess} -out={train_file}")

# 5. 合并 source1 和 source2 vcf
run_cmd(f"tabix -p vcf {output_dir}/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src1.biallelic.vcf.gz")
run_cmd(f"tabix -p vcf {output_dir}/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src2.biallelic.vcf.gz")
run_cmd(f"bcftools merge {output_dir}/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src1.biallelic.vcf.gz {output_dir}/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src2.biallelic.vcf.gz -Oz -o {merged_vcf}")
run_cmd(f"tabix -p vcf {merged_vcf}")

# 6. decode
run_cmd(f"hmmix decode -obs={obs_file_prefix}.{first_sample}.txt -mutrates={mutrates_file} -param={train_file} -admixpop={merged_vcf} -out={decode_output}")

print("\n🎉 全部完成！")
