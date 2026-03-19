import sys, os
ACC_DIR = "/home/linhuanyu/share1/20_AS3/results/inference/0_Scripts/1src/3_summary"
sys.path.append(ACC_DIR)
from acc import cal_accuracy_hap, cal_accuracy_sample, cal_accuracy_region
# 从命令行获取参数
demog = sys.argv[1]
nref = int(sys.argv[2])
ntgt = int(sys.argv[3])
seed = int(sys.argv[4])

print(f"demog={demog}, nref={nref}, ntgt={ntgt}, seed={seed}")

acc_out = f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker3.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/Infered_1src_Nov24.accuracy"


# # 先写表头（若不需要可去掉）
# with open(acc_out, "w") as f:
#     f.write("demog\tnref\tntgt\tseed\tcutoff\tlevel\tprecision\trecall\tf1\n")

infered_bed = f"/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker3.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/Infered_1src_src_Nov24.bed"
sim_bed     = f"/home/linhuanyu/share1/20_AS3/results/simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim1src.introgressed.tracts.bed"

hap_prec, hap_rec, hap_f1       = cal_accuracy_hap(sim_bed, infered_bed)              # 如需传 sample_col，请加参数
sample_prec, sample_rec, samp_f1= cal_accuracy_sample(sim_bed, infered_bed)
region_prec, region_rec, reg_f1 = cal_accuracy_region(sim_bed, infered_bed)

with open(acc_out, "a") as acc_f:  # 追加
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\thap\t{hap_prec}\t{hap_rec}\t{hap_f1}\n")
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\tsample\t{sample_prec}\t{sample_rec}\t{samp_f1}\n")
    acc_f.write(f"{demog}\t{nref}\t{ntgt}\t{seed}\tregion\t{region_prec}\t{region_rec}\t{reg_f1}\n")
