import sys
import os
import subprocess
import glob
import subprocess
import shutil

# 加载自己写的工具包路径
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import process_skovhmm_output_new, cal_accuracy

# 获取命令行参数
demog, nref, ntgt, seed = sys.argv[1:5]
nref = int(nref)
ntgt = int(ntgt)
seed = int(seed)

# 配置参数
threshold = 0.5
output_dir = "/home/linhuanyu/share1/20_AS3/results"

# SkovHMM输出目录
skovhmm_dir = os.path.join(output_dir, f"inference/SkovHMM/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")

# 文件路径
merged_hap_file = os.path.join(skovhmm_dir, "merged_hap.txt")
inferred_tracts_file = os.path.join(skovhmm_dir, "inferred_tracts.bed")
accuracy_file = os.path.join(skovhmm_dir, "accuracy.txt")

# 确保输出目录存在
os.makedirs(skovhmm_dir, exist_ok=True)

# 如果accuracy文件已经存在，则跳过
if os.path.exists(accuracy_file):
    print(f"⚡ {accuracy_file} already exists. Skipping...")
    sys.exit(0)

# 合并hap1和hap2文件
print(f"🔵 Merging hap1 and hap2 files into {merged_hap_file}...")
# 找到所有 hap1.txt 和 hap2.txt 文件
hap1_files = sorted(glob.glob(f"{skovhmm_dir}/*.hap1.txt"))
hap2_files = sorted(glob.glob(f"{skovhmm_dir}/*.hap2.txt"))

# 合并到一个列表
all_files = hap1_files + hap2_files

if len(all_files) == 0:
    raise FileNotFoundError(f"❌ 没有找到任何hap1或hap2文件在 {skovhmm_dir}，无法合并。")
elif len(all_files) == 1:
    # 只有一个文件，直接拷贝
    print(f"🔵 Only one file found. Copying {all_files[0]} to {merged_hap_file}")
    shutil.copy(all_files[0], merged_hap_file)
else:
    # 多个文件，使用cat合并
    files_to_merge = " ".join(all_files)
    merge_cmd = f"cat {files_to_merge} > {merged_hap_file}"
    print(f"🔵 Merging {len(all_files)} files into {merged_hap_file}")
    subprocess.run(merge_cmd, shell=True, check=True)

# 处理merged_hap.txt，输出推断tracts
print(f"🔵 Processing merged_hap.txt to inferred tracts...")
process_skovhmm_output_new(merged_hap_file, inferred_tracts_file, cutoff=threshold)

# 计算准确率
true_tracts_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim1src.introgressed.tracts.bed")
precision, recall = cal_accuracy(true_tracts_file, inferred_tracts_file)

# 写入accuracy
print(f"📝 Writing accuracy to {accuracy_file}...")
with open(accuracy_file, 'w') as f:
    f.write(f'{demog}\tnref_{nref}_ntgt_{ntgt}\t{threshold}\t{precision:.4f}\t{recall:.4f}\n')

print(f"🎯 全部完成！{demog}, nref={nref}, ntgt={ntgt}, seed={seed}, threshold={threshold}")
