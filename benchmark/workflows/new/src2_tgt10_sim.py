import sys
import os
import glob
import tskit
import pybedtools

# 加载自定义工具函数
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import get_introgressed_tracts

# 获取命令行参数
demog, nref, ntgt, seed = sys.argv[1:5]
nref = int(nref)
ntgt = int(ntgt)
seed = int(seed)

# 参数配置
output_dir = "/home/linhuanyu/share1/20_AS3/results/simulated_data"
sim_dir = os.path.join(output_dir, f"{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")
os.makedirs(sim_dir, exist_ok=True)

# 清理旧的 .bed 和 .bed.* 文件
for bed_file in glob.glob(os.path.join(sim_dir, "*.bed*")):
    try:
        os.remove(bed_file)
        print(f"[INFO] Removed old bed-related file: {bed_file}")
    except Exception as e:
        print(f"[WARNING] Failed to remove {bed_file}: {e}")

# 加载 tree sequence
ts_file = os.path.join(sim_dir, "sim2src.trees")
chr_name = "1"  # 注意：使用字符串以确保兼容 .bed 格式
ts = tskit.load(ts_file)
print(f"[INFO] Loaded tree sequence: {ts_file}")

# 定义输出文件名
src1_tracts_file = os.path.join(sim_dir, "sim2src.src1.introgressed.tracts.bed")
src2_tracts_file = os.path.join(sim_dir, "sim2src.src2.introgressed.tracts.bed")
src_tracts_file  = os.path.join(sim_dir, "sim2src.src.introgressed.tracts.bed")

# 处理不同的 demography 类型
if demog == 'HumanNeanderthalDenisovan':
    src1_id, src2_id, src3_id = "Nea1", "Den1", "Den2"
    tgt_id = "Papuan"

    den1_tracts_file = os.path.join(sim_dir, "sim.den1.introgressed.tracts.bed")
    den2_tracts_file = os.path.join(sim_dir, "sim.den2.introgressed.tracts.bed")

    get_introgressed_tracts(ts, chr_name, src1_id, tgt_id, output=src1_tracts_file)
    get_introgressed_tracts(ts, chr_name, src2_id, tgt_id, output=den1_tracts_file)
    get_introgressed_tracts(ts, chr_name, src3_id, tgt_id, output=den2_tracts_file)
    print(f"[INFO] Extracted tracts from {src1_id}, {src2_id}, {src3_id} → {tgt_id}")

    a = pybedtools.BedTool(den1_tracts_file)
    b = pybedtools.BedTool(den2_tracts_file)
    a.cat(b, postmerge=False).sort().saveas(src2_tracts_file)

    src1 = pybedtools.BedTool(src1_tracts_file)
    src2 = pybedtools.BedTool(src2_tracts_file)
    src1.cat(src2, postmerge=False).sort().saveas(src_tracts_file)

elif demog == 'ChimpBonoboGhost':
    src1_id, src2_id = "Ghost", "Bonobo"
    tgt_id = "Central"

    get_introgressed_tracts(ts, chr_name, src1_id, tgt_id, output=src1_tracts_file)
    get_introgressed_tracts(ts, chr_name, src2_id, tgt_id, output=src2_tracts_file)
    print(f"[INFO] Extracted tracts from {src1_id}, {src2_id} → {tgt_id}")

    src1 = pybedtools.BedTool(src1_tracts_file)
    src2 = pybedtools.BedTool(src2_tracts_file)
    src1.cat(src2, postmerge=False).sort().saveas(src_tracts_file)

elif demog == 'HumanArchaic':
    src1_id, src2_id = "ArchaicAFR", "Neanderthal"
    tgt_id = "CEU"

    get_introgressed_tracts(ts, chr_name, src1_id, tgt_id, output=src1_tracts_file)
    get_introgressed_tracts(ts, chr_name, src2_id, tgt_id, output=src2_tracts_file)
    print(f"[INFO] Extracted tracts from {src1_id}, {src2_id} → {tgt_id}")

    src1 = pybedtools.BedTool(src1_tracts_file)
    src2 = pybedtools.BedTool(src2_tracts_file)
    src1.cat(src2, postmerge=False).sort().saveas(src_tracts_file)

elif demog == 'AS2_HumanNeanderthalDenisovan':
    src1_id, src2_id = "Nean2", "Den1"
    tgt_id = "Europe"

    get_introgressed_tracts(ts, chr_name, src1_id, tgt_id, output=src1_tracts_file)
    get_introgressed_tracts(ts, chr_name, src2_id, tgt_id, output=src2_tracts_file)
    print(f"[INFO] Extracted tracts from {src1_id}, {src2_id} → {tgt_id}")

    src1 = pybedtools.BedTool(src1_tracts_file)
    src2 = pybedtools.BedTool(src2_tracts_file)
    src1.cat(src2, postmerge=False).sort().saveas(src_tracts_file)

else:
    raise ValueError(f"[ERROR] Unrecognized demog: {demog}")

# 最终日志输出
print(f"[INFO] src1 tracts file: {src1_tracts_file}")
print(f"[INFO] src2 tracts file: {src2_tracts_file}")
print(f"[INFO] Final merged tracts written to: {src_tracts_file}")
