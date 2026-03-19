import sys
import os
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import process_archaicseeker3_output, process_archaicseeker3_1src_output, cal_accuracy

demog, nref, ntgt, seed = sys.argv[1:5]
nref = int(nref)
ntgt = int(ntgt)
seed = int(seed)

output_dir = "/home/linhuanyu/share1/20_AS3/results"  # ← 请替换成你 config.yaml 中的 output_dir
prefix = os.path.join(output_dir, f"inference/ArchaicSeeker3.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")
seg_file = os.path.join(prefix, "AS3_Mamba_Smoother_4096_2048_0.bed")
bed_out_1 = os.path.join(prefix, f"AS3_Mamba_Smoother_4096_2048_0.out.src1.bed")
bed_out_2 = os.path.join(prefix, f"AS3_Mamba_Smoother_4096_2048_0.out.src2.bed")
bed_out = os.path.join(prefix, f"AS3_Mamba_Smoother_4096_2048_0.out.bed")
acc_out = os.path.join(prefix, f"AS3_Mamba_Smoother_4096_2048_0.out.accuracy")
true_file_1 = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src1.introgressed.tracts.bed")
true_file_2 = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src2.introgressed.tracts.bed")
merged_true_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.introgressed.tracts.bed")

os.makedirs(os.path.dirname(bed_out), exist_ok=True)

with open(merged_true_file, 'w') as fout_merge:
    for source_file in [true_file_1, true_file_2]:
        with open(source_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    fout_merge.write("\t".join(parts[:3]) + "\n")

process_archaicseeker3_output(seg_file, bed_out_1, bed_out_2, 0.4)
process_archaicseeker3_1src_output(seg_file, bed_out, 0.4)

p11, r11 = cal_accuracy(true_file_1, bed_out_1)
p22, r22 = cal_accuracy(true_file_2, bed_out_2)
p12, r12 = cal_accuracy(true_file_1, bed_out_2)
p21, r21 = cal_accuracy(true_file_2, bed_out_1)
precision, recall = cal_accuracy(merged_true_file, bed_out)

# 定义F1函数
def f1(p, r):
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)

# 计算每个单独的 F1
f1_11 = f1(p11, r11)
f1_22 = f1(p22, r22)
f1_12 = f1(p12, r12)
f1_21 = f1(p21, r21)

# 判断逻辑
# 方案1：bed_out_1 -> src1, bed_out_2 -> src2
# 方案2：bed_out_2 -> src1, bed_out_1 -> src2

# 总F1
f1_pair1 = f1_11 + f1_22
f1_pair2 = f1_12 + f1_21

# 方案1 四个指标的均值
mean1 = np.mean([p11, r11, p22, r22])
# 方案2 四个指标的均值
mean2 = np.mean([p12, r12, p21, r21])

# 先选总F1高的，如果总F1差不多，再比较四个指标的平均值
if abs(f1_pair1 - f1_pair2) > 0.01:
    # F1差异大，选F1高
    if f1_pair1 >= f1_pair2:
        precision_1, recall_1 = p11, r11
        precision_2, recall_2 = p22, r22
    else:
        precision_1, recall_1 = p12, r12
        precision_2, recall_2 = p21, r21
else:
    # F1差异小，选四个指标综合均值高
    if mean1 >= mean2:
        precision_1, recall_1 = p11, r11
        precision_2, recall_2 = p22, r22
    else:
        precision_1, recall_1 = p12, r12
        precision_2, recall_2 = p21, r21

with open(acc_out, 'w') as f:
    f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc1\t0.4\t{precision_1:.2f}\t{recall_1:.2f}\n")
    f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc2\t0.4\t{precision_2:.2f}\t{recall_2:.2f}\n")
    f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc\t0.4\t{precision:.2f}\t{recall:.2f}\n")
