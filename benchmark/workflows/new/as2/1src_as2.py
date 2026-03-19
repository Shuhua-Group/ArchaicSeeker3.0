import os
import sys
import subprocess
from pathlib import Path

# 添加 utils 路径
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import process_as2_1src_output, cal_accuracy

# 读取参数
demog, nref, ntgt, seed = sys.argv[1:5]
nref = int(nref)
ntgt = int(ntgt)
seed = int(seed)

# 配置路径
output_dir = "/home/linhuanyu/share1/20_AS3/results"
as2_exec = "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/ext/ArchaicSeeker2/archaicseeker2"  # 修改为你的路径
prefix = os.path.join(output_dir, f"inference/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")
output_prefix = os.path.join(prefix, "archaicseeker2.out")
accuracy_file = output_prefix + ".accuracy"

# 如果结果已存在，跳过
if os.path.exists(accuracy_file):
    print(f"⚡ {accuracy_file} already exists. Skipping...")
    sys.exit(0)

# 创建输出目录
os.makedirs(prefix, exist_ok=True)

# 处理 VCF：拆分 ref、tgt、src1、src2，并写 par 文件
print("🚀 Processing VCF files...")

vcf_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim1src.biallelic.dup.vcf.gz")
ref_list = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/sim.ref.list"
tgt_list = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/sim.tgt.list"
src1_list = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/sim.src1.list"
src2_list = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/sim.src2.list"

ref_vcf = os.path.join(prefix, "sim.ref.vcf.gz")
tgt_vcf = os.path.join(prefix, "sim.tgt.vcf.gz")
src1_vcf = os.path.join(prefix, "sim.src1.vcf.gz")
src2_vcf = os.path.join(prefix, "sim.src2.vcf.gz")
vcf_par = os.path.join(prefix, "sim.vcf.par")

Path(ref_vcf).parent.mkdir(parents=True, exist_ok=True)

cmds = [
    (ref_list, ref_vcf),
    (tgt_list, tgt_vcf),
    (src1_list, src1_vcf),
    (src2_list, src2_vcf),
]
for sample_list, out_vcf in cmds:
    subprocess.run(f"bcftools view {vcf_file} -S {sample_list} | bgzip -c > {out_vcf}", shell=True, check=True)

with open(vcf_par, 'w') as f:
    f.write("vcf\n")
    f.write(f"{ref_vcf}\n")
    f.write(f"{tgt_vcf}\n")
    f.write(f"{src1_vcf}\n")
    f.write(f"{src2_vcf}\n")

print("✅ VCF processing complete.")

# 运行 archaicseeker2
print("🚀 Running ArchaicSeeker2...")

remap_par = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/remap.par"
model = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/model.txt"
outgroup_par = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/outgroup.par"
pop_par = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/sim.pop.par"
anc_par = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/anc.par"

cmd = (
    f"{as2_exec} -v {vcf_par} -r {remap_par} -m {model} "
    f"-X {outgroup_par} -p {pop_par} -A {anc_par} -o {output_prefix}"
)

try:
    subprocess.run(cmd, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ ArchaicSeeker2执行失败: {e}")
    sys.exit(1)

print("✅ ArchaicSeeker2执行完成，开始处理输出...")

# 处理输出结果
seg_file = output_prefix + ".seg"
true_tracts_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim1src.introgressed.tracts.bed")
inferred_tracts_file = output_prefix + ".bed"

process_as2_1src_output(seg_file, inferred_tracts_file)

# 计算准确率
precision, recall = cal_accuracy(true_tracts_file, inferred_tracts_file)

# 写入 accuracy
with open(accuracy_file, 'w') as f:
    f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\t-1\t{precision:.4f}\t{recall:.4f}\n")

print(f"🎯 全部完成！{demog}, nref={nref}, ntgt={ntgt}, seed={seed}")
