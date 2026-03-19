import sys
import os
import subprocess
import pandas as pd

# 加载自己写的工具包
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")

# 解析命令行参数
demog, nref, ntgt, seed = sys.argv[1:5]
nref = int(nref)
ntgt = int(ntgt)
seed = int(seed)

# 配置参数
threshold = 50000
output_dir = "/home/linhuanyu/share1/20_AS3/results"
sprime_exec = "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/ext/SPrime/sprime.jar"
map_arch_exec = "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/ext/SPrime/sprimepipeline/pub.pipeline.pbs/tools/map_arch_genome/map_arch"
score_summary_exec = "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/ext/SPrime/sprimepipeline/pub.pipeline.pbs/tools/score_summary.r"

# 根据demog设置mu
mu = 1.4e-8 if demog == 'HumanNeanderthalDenisovan' else 1.2e-8

# 输入文件路径
gt_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.biallelic.vcf.gz")
outgroup_file = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/SPrime/{demog}/nref_{nref}/sim.outgroup.ids"
map_file = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/SPrime/{demog}/sim.map"
exsamps_file = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/SPrime/{demog}/nref_{nref}/ntgt_{ntgt}/sim.excluded.ids"

# 输出目录和前缀
prefix = os.path.join(output_dir, f"inference/Sprime/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")
output_prefix = os.path.join(prefix, "sprime.2src.out")
accuracy_file = output_prefix + ".accuracy"

# 如果accuracy已经存在，跳过
if os.path.exists(accuracy_file):
    print(f"⚡ {accuracy_file} already exists. Skipping...")
    sys.exit(0)

# 确保输出目录存在
os.makedirs(prefix, exist_ok=True)

# SPrime命令
cmd_sprime = (
    f"java -Xmx10g -jar {sprime_exec} "
    f"gt={gt_file} outgroup={outgroup_file} map={map_file} "
    f"out={output_prefix} minscore={threshold} mu={mu} excludesamples={exsamps_file}"
)

print(f"🚀 Running SPrime command:\n{cmd_sprime}")
try:
    subprocess.run(cmd_sprime, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ SPrime执行失败: {e}")
    sys.exit(1)

print("✅ SPrime执行完成，开始匹配源群...")

# 匹配src1和src2
score_file = output_prefix + ".score"
src1_vcf = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src1.biallelic.vcf.gz")
src2_vcf = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src2.biallelic.vcf.gz")

src1_score_tmp = output_prefix + ".src1.mscore.tmp"
src1_score_final = output_prefix + ".src1.mscore"
src2_score_tmp = output_prefix + ".src2.mscore"
match_rate_output = output_prefix + ".match.rate"

# 运行 map_arch for src1
cmd_src1 = f"{map_arch_exec} --kpall --vcf {src1_vcf} --score {score_file} --tag src1 --sep '\\t' > {src1_score_tmp}"
print(f"🚀 Mapping src1:\n{cmd_src1}")
subprocess.run(cmd_src1, shell=True, check=True)

# 检查src1输出是否为空
if os.path.exists(src1_score_tmp) and os.path.getsize(src1_score_tmp) > 0:
    os.rename(src1_score_tmp, src1_score_final)
else:
    print(f"⚠️ 警告: src1匹配为空，填充NA结果")
    with open(match_rate_output, 'w') as f:
        f.write("chr\tseg\tfrom\tto\tsrc1\tsrc2\nNA\tNA\tNA\tNA\tNA\tNA\n")
    sys.exit(0)

# 运行 map_arch for src2
cmd_src2 = f"{map_arch_exec} --kpall --vcf {src2_vcf} --score {src1_score_final} --tag src2 --sep '\\t' > {src2_score_tmp}"
print(f"🚀 Mapping src2:\n{cmd_src2}")
subprocess.run(cmd_src2, shell=True, check=True)

# 删掉src1.mscore
os.remove(src1_score_final)

# 运行R脚本总结match rate
cmd_rscript = f"Rscript {score_summary_exec} {prefix} {match_rate_output}"
print(f"🚀 Summarizing match rate:\n{cmd_rscript}")
subprocess.run(cmd_rscript, shell=True, check=True)

print("✅ 匹配完成，开始处理预测区段...")
