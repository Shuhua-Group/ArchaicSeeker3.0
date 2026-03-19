#!/bin/bash
#SBATCH --job-name=hmmix_stat_1src_chunk1
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --mem=5G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-999  # 这里根据实际任务数，比如999表示1000个任务（0到999）

# 加载环境
source ~/.bashrc
conda activate sstar-analysis  # 换成你的conda环境名

# 切换到日志目录（如果没有可以删掉下面两行）
mkdir -p /home/linhuanyu/00_log/05_AS3/2504/hmmix
cd /home/linhuanyu/00_log/05_AS3/2504/hmmix

# 准备参数列表文件
TASK_LIST="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/task_1src_chunks/task_1src_chunk1.txt"

# 取出本任务对应的行
task=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")

# 分割成四个参数
demog=$(echo $task | awk '{print $1}')
nref=$(echo $task | awk '{print $2}')
ntgt=$(echo $task | awk '{print $3}')
seed=$(echo $task | awk '{print $4}')

# 执行你的Python脚本
python /home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/1src_hmmix_stat.py "$demog" "$nref" "$ntgt" "$seed"
