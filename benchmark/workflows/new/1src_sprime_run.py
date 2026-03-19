import sys
import os
import subprocess

# 加载自己写的工具包路径
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import process_sprime_output, cal_accuracy

# 获取命令行参数
demog, nref, ntgt, seed = sys.argv[1:5]
nref = int(nref)
ntgt = int(ntgt)
seed = int(seed)

# 配置参数
threshold = 50000
output_dir = "/home/linhuanyu/share1/20_AS3/results"
sprime_exec = "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/ext/SPrime/sprime.jar"  # SPrime程序路径

# 根据demog设置mu
mu = 1.29e-8 if demog == 'HumanNeanderthal' else 1.2e-8

# 输入文件路径
gt_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim1src.biallelic.vcf.gz")
outgroup_file = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/SPrime/{demog}/sim.{nref}.outgroup.ids"
map_file = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/SPrime/{demog}/sim.map"

# 输出路径
prefix = os.path.join(output_dir, f"inference/SPrime/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")
output_prefix = os.path.join(prefix, "sprime.1src.out")
accuracy_file = output_prefix + ".accuracy"

# 检查 accuracy 文件是否已存在，存在就跳过
if os.path.exists(accuracy_file):
    print(f"⚡ {accuracy_file} already exists. Skipping...")
    sys.exit(0)

# 确保输出目录存在
os.makedirs(prefix, exist_ok=True)

# 构建命令
cmd = (
    f"java -Xmx5g -jar {sprime_exec} "
    f"gt={gt_file} outgroup={outgroup_file} map={map_file} "
    f"out={output_prefix} minscore={threshold} mu={mu}"
)

# 打印并执行
print(f"🚀 Running command:\n{cmd}")

try:
    subprocess.run(cmd, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ SPrime执行失败: {e}")
    sys.exit(1)

print("✅ SPrime执行完成，开始处理输出...")

# 定义处理阶段的输入输出文件
scores_file = output_prefix + ".score"
true_tracts_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim1src.introgressed.tracts.bed")
inferred_tracts_file = output_prefix + ".bed"

# 处理sprime输出
process_sprime_output(scores_file, inferred_tracts_file)

# 计算准确率
precision, recall = cal_accuracy(true_tracts_file, inferred_tracts_file)

# 写入accuracy文件
with open(accuracy_file, 'w') as f:
    f.write(f'{demog}\tnref_{nref}_ntgt_{ntgt}\t{threshold}\t{precision:.4f}\t{recall:.4f}\n')

print(f"🎯 全部完成！{demog}, nref={nref}, ntgt={ntgt}, seed={seed}, threshold={threshold}")
