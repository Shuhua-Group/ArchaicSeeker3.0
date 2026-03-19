import os
import sys
import numpy as np
import subprocess
from pathlib import Path

# 添加 utils 路径（如有分析函数后处理可启用）
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")

# 获取参数
demog, nref, ntgt, seed = sys.argv[1:5]
nref = int(nref)
ntgt = int(ntgt)
seed = int(seed)

# 配置路径
output_dir = "/home/linhuanyu/share1/20_AS3/results"
as2_exec = "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/ext/ArchaicSeeker2.0/libnlopt.so.0_free/ArchaicSeeker2"
prefix = os.path.join(output_dir, f"inference/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")
output_prefix = os.path.join(prefix, "archaicseeker2.out")

# 创建目录
os.makedirs(prefix, exist_ok=True)

# ==== 1. 处理 VCF ====
print("🚀 [1/3] Processing VCF for 2src...")

vcf_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim2src.biallelic.vcf.gz")
ref_list  = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/sim.ref.list"
tgt_list  = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/sim.tgt.list"
src1_list = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/sim.src1.list"
src2_list = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/sim.src2.list"

ref_vcf  = os.path.join(prefix, "sim.ref.vcf.gz")
tgt_vcf  = os.path.join(prefix, "sim.tgt.vcf.gz")
src1_vcf = os.path.join(prefix, "sim.src1.vcf.gz")
src2_vcf = os.path.join(prefix, "sim.src2.vcf.gz")
vcf_par  = os.path.join(prefix, "sim.vcf.par")

Path(ref_vcf).parent.mkdir(parents=True, exist_ok=True)

# 拆分 VCF
cmds = [
    (ref_list, ref_vcf),
    (tgt_list, tgt_vcf),
    (src1_list, src1_vcf),
    (src2_list, src2_vcf),
]
for sample_list, out_vcf in cmds:
    subprocess.run(f"bcftools view {vcf_file} -S {sample_list} | bgzip -c > {out_vcf}", shell=True, check=True)

# 写 par 文件
with open(vcf_par, 'w') as f:
    f.write("vcf\n")
    f.write(f"{ref_vcf}\n{tgt_vcf}\n{src1_vcf}\n{src2_vcf}\n")

print("✅ VCF + .par done.")

# ==== 2. 运行 AS2 ====
print("🚀 [2/3] Running ArchaicSeeker2.0...")

remap_par   = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/remap.par"
model_file  = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/model.txt"
outgroup_par = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/outgroup.par"
pop_par     = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/nref_{nref}/ntgt_{ntgt}/sim.pop.par"
anc_par     = f"/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/ArchaicSeeker2.0/{demog}/anc.par"

cmd = (
    f"{as2_exec} -v {vcf_par} -r {remap_par} -m {model_file} "
    f"-X {outgroup_par} -p {pop_par} -A {anc_par} -o {output_prefix}"
)

try:
    subprocess.run(cmd, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ ArchaicSeeker2 failed: {e}")
    sys.exit(1)

print("✅ ArchaicSeeker2 finished.")

print(f"🎯 Done! {demog} | nref={nref}, ntgt={ntgt}, seed={seed}")
