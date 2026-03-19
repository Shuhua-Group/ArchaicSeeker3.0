#!/bin/bash
#SBATCH --job-name=sprime
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=6G
#SBATCH --partition=batch

# 参数
demog=$1
nref=$2
ntgt=$3
seed=$4

# 日志
echo "🚀 Running SPrime 1src pipeline..."
echo "Demography: $demog, Nref: $nref, Ntgt: $ntgt, Seed: $seed"

# 加载环境
source ~/.bashrc
conda activate sstar-analysis

# 运行 Python 脚本
python /home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/1src_sprime_run.py "$demog" "$nref" "$ntgt" "$seed"

# 结束
echo "✅ Finished SPrime 1src for $demog Nref=$nref Ntgt=$ntgt Seed=$seed"