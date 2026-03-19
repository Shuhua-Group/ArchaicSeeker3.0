import sys, os
ACC_DIR = "/home/linhuanyu/share1/20_AS3/0_Scripts/2src/3_summary"
sys.path.append(ACC_DIR)
from acc import cal_accuracy_hap, cal_accuracy_sample, cal_accuracy_region
# 从命令行获取参数
demog = sys.argv[1]
nref = int(sys.argv[2])
ntgt = int(sys.argv[3])
seed = int(sys.argv[4])

print(f"demog={demog}, nref={nref}, ntgt={ntgt}, seed={seed}")

acc_out = f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/Infered_2src.accuracy"
if os.path.exists(acc_out):
    os.remove(acc_out)
# # 先写表头（若不需要可去掉）
# with open(acc_out, "w") as f:
#     f.write("demog\tnref\tntgt\tseed\tcutoff\tlevel\tprecision\trecall\tf1\n")

infered_src_bed = f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/Infered_2src_src.bed"
infered_src1_bed = f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/Infered_2src_src1.bed"
infered_src2_bed = f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/Infered_2src_src2.bed"

sim_src_bed     = f"/home/linhuanyu/share1/20_AS3/results/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src.introgressed.tracts.bed"
sim_src1_bed = f"/home/linhuanyu/share1/20_AS3/results/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src1.introgressed.tracts.bed"
sim_src2_bed = f"/home/linhuanyu/share1/20_AS3/results/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.src2.introgressed.tracts.bed"

src_hap_prec, src_hap_rec, src_hap_f1       = cal_accuracy_hap(sim_src_bed, infered_src_bed)              # 如需传 sample_col，请加参数
src_sample_prec, src_sample_rec, src_sample_f1= cal_accuracy_sample(sim_src_bed, infered_src_bed)
src_region_prec, src_region_rec, src_region_f1 = cal_accuracy_region(sim_src_bed, infered_src_bed)

src1_hap_prec, src1_hap_rec, src1_hap_f1       = cal_accuracy_hap(sim_src1_bed, infered_src1_bed)              # 如需传 sample_col，请加参数\\
src1_sample_prec, src1_sample_rec, src1_sample_f1= cal_accuracy_sample(sim_src1_bed, infered_src1_bed)
src1_region_prec, src1_region_rec, src1_region_f1 = cal_accuracy_region(sim_src1_bed, infered_src1_bed)

src2_hap_prec, src2_hap_rec, src2_hap_f1       = cal_accuracy_hap(sim_src2_bed, infered_src2_bed)              # 如需传 sample_col，请加参数\\
src2_sample_prec, src2_sample_rec, src2_sample_f1= cal_accuracy_sample(sim_src2_bed, infered_src2_bed)
src2_region_prec, src2_region_rec, src2_region_f1 = cal_accuracy_region(sim_src2_bed, infered_src2_bed)
with open(acc_out, "a") as acc_f:  # 追加
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\tsrc\thap\t{src_hap_prec}\t{src_hap_rec}\t{src_hap_f1}\n")
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\tsrc\tsample\t{src_sample_prec}\t{src_sample_rec}\t{src_sample_f1}\n")
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\tsrc\tregion\t{src_region_prec}\t{src_region_rec}\t{src_region_f1}\n")
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\tsrc1\thap\t{src1_hap_prec}\t{src1_hap_rec}\t{src1_hap_f1}\n")
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\tsrc1\tsample\t{src1_sample_prec}\t{src1_sample_rec}\t{src1_sample_f1}\n")
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\tsrc1\tregion\t{src1_region_prec}\t{src1_region_rec}\t{src1_region_f1}\n")
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\tsrc2\thap\t{src2_hap_prec}\t{src2_hap_rec}\t{src2_hap_f1}\n")
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\tsrc2\tsample\t{src2_sample_prec}\t{src2_sample_rec}\t{src2_sample_f1}\n")
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\tsrc2\tregion\t{src2_region_prec}\t{src2_region_rec}\t{src2_region_f1}\n")

