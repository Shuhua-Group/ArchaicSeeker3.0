#!/bin/bash
#SBATCH --job-name=run
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
python /home/linhuanyu/share1/20_AS3/0_Scripts/1src/1_run/sprime.py "$demog" "$nref" "$ntgt" "$seed"
# python /home/linhuanyu/share1/20_AS3/results/inference/0_Scripts/1src/1_run/as2.py "$demog" "$nref" "$ntgt" "$seed"

# 结束
echo "✅ Finished SPrime 1src for $demog Nref=$nref Ntgt=$ntgt Seed=$seed"