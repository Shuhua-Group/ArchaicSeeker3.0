import sys
import os
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import process_archaicseeker3_output, process_archaicseeker3_1src_output, cal_accuracy

demog, nref, ntgt, seed, cutoff = sys.argv[1:6]
nref = int(nref)
ntgt = int(ntgt)
seed = int(seed)
cutoff = float(cutoff)

output_dir = "/home/linhuanyu/share1/20_AS3/results"  # ← 请替换成你 config.yaml 中的 output_dir
prefix = os.path.join(output_dir, f"inference/ArchaicSeeker3.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")
seg_file = os.path.join(prefix, "AS3_Mamba_Smoother_4096_2048_5216.bed")
bed_out_1 = os.path.join(prefix, f"AS3_Mamba_Smoother_4096_2048_5216.out.cutoff.{cutoff}.src1.bed")
bed_out_2 = os.path.join(prefix, f"AS3_Mamba_Smoother_4096_2048_5216.out.cutoff.{cutoff}.src2.bed")
bed_out = os.path.join(prefix, f"AS3_Mamba_Smoother_4096_2048_5216.out.cutoff.{cutoff}.bed")
acc_out = os.path.join(prefix, f"AS3_Mamba_Smoother_4096_2048_5216.out.cutoff.{cutoff}.accuracy")
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

process_archaicseeker3_output(seg_file, bed_out_1, bed_out_2, cutoff)
process_archaicseeker3_1src_output(seg_file, bed_out, cutoff)

p11, r11 = cal_accuracy(true_file_1, bed_out_1)
p22, r22 = cal_accuracy(true_file_2, bed_out_2)
p12, r12 = cal_accuracy(true_file_1, bed_out_2)
p21, r21 = cal_accuracy(true_file_2, bed_out_1)
precision, recall = cal_accuracy(merged_true_file, bed_out)

if r11 + r22 >= r12 + r21:
    precision_1, recall_1 = p11, r11
    precision_2, recall_2 = p22, r22
else:
    precision_1, recall_1 = p12, r12
    precision_2, recall_2 = p21, r21

with open(acc_out, 'w') as f:
    f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc1\t{cutoff}\t{precision_1:.2f}\t{recall_1:.2f}\n")
    f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc2\t{cutoff}\t{precision_2:.2f}\t{recall_2:.2f}\n")
    f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\tsrc\t{cutoff}\t{precision:.2f}\t{recall:.2f}\n")
