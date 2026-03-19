import sys
import os
sys.path.insert(0, "/home/linhuanyu/83_AS3_SSTAR/sstar-analysis")
from utils.utils import process_ibdmix_output, cal_accuracy_tgt1, cal_accuracy_tgt10
import pandas as pd

# 获取命令行参数
demog, nref, ntgt, seed = sys.argv[1:6]
nref = int(nref)
ntgt = int(ntgt)
seed = int(seed)
cutoff = 4

# 输出路径配置
output_dir = "/home/linhuanyu/share1/20_AS3/results"
prefix = os.path.join(output_dir, f"inference/IBDmix/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}")

seg_file = os.path.join(prefix, "ibdmix_output.txt")
true_file = os.path.join(output_dir, f"simulated_data/{demog}/nref_{nref}/ntgt_{ntgt}/{seed}/sim1src.introgressed.tracts.bed")

bed_out = os.path.join(prefix, f"IBDmix.out.cutoff.{cutoff}.bed")
acc_out = os.path.join(prefix, f"IBDmix.out.cutoff.{cutoff}.accuracy")

# 检查段文件是否存在
if not os.path.exists(seg_file):
    print(f"[SKIP] {prefix} 缺失 segment 文件，跳过。")
    sys.exit(0)

# 执行处理
os.makedirs(os.path.dirname(bed_out), exist_ok=True)
process_ibdmix_output(seg_file, bed_out, cutoff)

precision, recall, f1, _ = cal_accuracy_tgt10(true_file, bed_out)

# 写入主评估文件
with open(acc_out, 'w') as f:
    f.write(f"{demog}\tnref_{nref}_ntgt_{ntgt}\t{cutoff}\t{precision:.2f}\t{recall:.2f}\t{f1:.2f}\n")

print(f"✅ 成功处理: {demog}, nref={nref}, ntgt={ntgt}, seed={seed}, cutoff={cutoff}")