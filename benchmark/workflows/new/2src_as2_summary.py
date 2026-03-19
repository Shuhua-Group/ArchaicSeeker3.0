import os
import subprocess
import sys

# 加载 utils
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import cal_accuracy

# 传入参数
demog, nref, ntgt, seed = sys.argv[1:5]
nref = int(nref)
ntgt = int(ntgt)
seed = int(seed)

output_dir = "/home/linhuanyu/share1/20_AS3/results"
threshold = -1
prefix = os.path.join(output_dir, f"inference/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")
output_prefix = os.path.join(prefix, "archaicseeker2.out")
accuracy_file = output_prefix + ".0505.accuracy"

# 文件路径
src1_true_tracts = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src1.introgressed.tracts.bed")
src2_true_tracts = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src2.introgressed.tracts.bed")
src1_inferred_tracts = output_prefix + ".src1.bed"
src2_inferred_tracts = output_prefix + ".src2.bed"

# 分别计算 src1, src2
src1_precision, src1_recall = cal_accuracy(src1_true_tracts, src1_inferred_tracts)
src2_precision, src2_recall = cal_accuracy(src2_true_tracts, src2_inferred_tracts)

# 定义输出文件路径
true_merged_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}", f"sim2src.introgressed.tracts.bed")
inferred_merged_file = os.path.join(output_dir, f"inference/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/src.bed")

# 确保目录存在
os.makedirs(os.path.dirname(true_merged_file), exist_ok=True)

# 合并 true
subprocess.run(f"cat {src1_true_tracts} {src2_true_tracts} | cut -f1-3 > {true_merged_file}", shell=True, check=True)
subprocess.run(f"cat {src1_inferred_tracts} {src2_inferred_tracts} | cut -f1-3 > {inferred_merged_file}", shell=True, check=True)


# 计算 merged
precision, recall = cal_accuracy(true_merged_file, inferred_merged_file)

# 输出
print(f"src1_precision: {src1_precision:.4f}, src1_recall: {src1_recall:.4f}")
print(f"src2_precision: {src2_precision:.4f}, src2_recall: {src2_recall:.4f}")
print(f"merged_precision: {precision:.4f}, merged_recall: {recall:.4f}")

# 可选：写入文件
# 删除旧文件（如果存在）
if os.path.exists(accuracy_file):
    os.remove(accuracy_file)

# 写入新结果
with open(accuracy_file, 'w') as f:
    f.write(f'{demog}\tnref_{nref}_ntgt_{ntgt}\t{threshold}\tsrc1\t{src1_precision:.4f}\t{src1_recall:.4f}\n')
    f.write(f'{demog}\tnref_{nref}_ntgt_{ntgt}\t{threshold}\tsrc2\t{src2_precision:.4f}\t{src2_recall:.4f}\n')
    f.write(f'{demog}\tnref_{nref}_ntgt_{ntgt}\t{threshold}\tsrc\t{precision:.4f}\t{recall:.4f}\n')


print(f"✅ Accuracy written to {accuracy_file}")
