#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=1G
#SBATCH --partition=batch

# 参数
demog=$1
nref=$2
ntgt=$3
seed=$4


# 加载环境
source ~/.bashrc
conda activate sstar-analysis

# 运行 Python 脚本
# python /home/linhuanyu/share1/20_AS3/0_Scripts/2src/3_summary/as2.py "$demog" "$nref" "$ntgt" "$seed"
# python /home/linhuanyu/share1/20_AS3/0_Scripts/2src/3_summary/as3.py "$demog" "$nref" "$ntgt" "$seed"
python /home/linhuanyu/share1/20_AS3/0_Scripts/2src/3_summary/IBDmix.py "$demog" "$nref" "$ntgt" "$seed"
# python /home/linhuanyu/share1/20_AS3/0_Scripts/2src/3_summary/HMMix.py "$demog" "$nref" "$ntgt" "$seed"
# python /home/linhuanyu/share1/20_AS3/0_Scripts/2src/3_summary/Sprime.py "$demog" "$nref" "$ntgt" "$seed"
