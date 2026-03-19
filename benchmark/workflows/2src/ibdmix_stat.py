import sys
import os
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import process_ibdmix_output, cal_accuracy_tgt1, cal_accuracy_tgt10
import subprocess

# 获取命令行参数
demog, nref, ntgt, seed= sys.argv[1:5]
nref = int(nref)
ntgt = int(ntgt)
seed = int(seed)
cutoff = 4

# 输出路径配置
output_dir = "/home/linhuanyu/share1/20_AS3/results"
prefix = os.path.join(output_dir, f"inference/IBDmix/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")

seg_file_1 = os.path.join(prefix, "ibdmix_arc1_output.txt")
seg_file_2 = os.path.join(prefix, "ibdmix_arc2_output.txt")
true_file_1 = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src1.introgressed.tracts.bed")
true_file_2 = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src2.introgressed.tracts.bed")

bed_out_1 = os.path.join(prefix, f"IBDmix.out.src1.cutoff.{cutoff}.bed")
bed_out_2 = os.path.join(prefix, f"IBDmix.out.src2.cutoff.{cutoff}.bed")
acc_out = os.path.join(prefix, f"IBDmix.out.cutoff.{cutoff}.accuracy")

# 检查段文件是否存在
if not os.path.exists(seg_file_1) or not os.path.exists(seg_file_2):
    print(f"[SKIP] {prefix} 缺失 segment 文件，跳过。")
    sys.exit(0)

# 执行处理
os.makedirs(os.path.dirname(bed_out_1), exist_ok=True)
os.makedirs(os.path.dirname(bed_out_2), exist_ok=True)
process_ibdmix_output(seg_file_1, bed_out_1, cutoff)
process_ibdmix_output(seg_file_2, bed_out_2, cutoff)

precision_1, recall_1,f11,_ = cal_accuracy_tgt10(true_file_1, bed_out_1)
precision_2, recall_2,f12,_ = cal_accuracy_tgt10(true_file_2, bed_out_2)

merged_true_file = os.path.join(prefix, f"merged_true_tracts_{demog}_nref{nref}_ntgt{ntgt}_seed{seed}.bed")
subprocess.run(f"cat {true_file_1} {true_file_2} | cut -f1-4 > {merged_true_file}", shell=True, check=True)

# 合并 inferred tracts
merged_bed_out = os.path.join(prefix, f"merged_inferred_tracts_cutoff{cutoff}.bed")
subprocess.run(f"cat {bed_out_1} {bed_out_2} | cut -f1-3 > {merged_bed_out}", shell=True, check=True)

# 计算 merged accuracy
precision, recall,f1,_ = cal_accuracy_tgt10(merged_true_file, merged_bed_out)


# 写入主评估文件
with open(acc_out, 'w') as f:
    f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc1\t{cutoff}\t{precision_1:.2f}\t{recall_1:.2f}\t{f11:.2f}\n")
    f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc2\t{cutoff}\t{precision_2:.2f}\t{recall_2:.2f}\t{f12:.2f}\n")
    f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc\t{cutoff}\t{precision:.2f}\t{recall:.2f}\t{f1:.2f}\n")
print(f"✅ 成功处理: {demog}, nref={nref}, ntgt={ntgt}, seed={seed}, cutoff={cutoff}")
