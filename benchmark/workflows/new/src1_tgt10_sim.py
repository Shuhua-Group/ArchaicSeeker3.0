import sys
import os
import tskit

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

ts_file = os.path.join(sim_dir, "sim1src.trees")
tracts_file = os.path.join(sim_dir, "sim1src.introgressed.tracts.bed")
chr_name = "1"  # 输出为字符串更通用

# 人口模型匹配
if demog == 'BonoboGhost':
    src_id = "Ghost"
    tgt_id = "Bonobo"
elif demog == 'HumanNeanderthal':
    src_id = "Nea"
    tgt_id = "CEU"
elif demog == 'OOANeanderthal':
    src_id = "NEA"
    tgt_id = "CEU"
elif demog == 'AncientEurasia':
    src_id = "Neanderthal"
    tgt_id = "Loschbour"
else:
    raise ValueError(f"Unrecognized demog: {demog}")

# 日志输出
print(f"[INFO] Loading tree sequence from: {ts_file}")
print(f"[INFO] Extracting tracts: {src_id} → {tgt_id}")
print(f"[INFO] Writing to: {tracts_file}")

# 加载 tree sequence 并提取
ts = tskit.load(ts_file)
get_introgressed_tracts(ts, chr_name=chr_name, src_name=src_id, tgt_name=tgt_id, output=tracts_file)
