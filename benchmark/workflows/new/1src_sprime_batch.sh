#!/bin/bash
#SBATCH --job-name=sprime
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=6G
#SBATCH --partition=batch

# 任务列表
task_file="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/task_1src_list.txt"

# 运行脚本
sprime_py="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/1src_sprime_run.py"

# 切换到日志目录
cd /home/linhuanyu/00_log/05_AS3/2504/Sprime

# 逐行读取任务
while read -r demog nref ntgt seed; do
  jobname="sprime_${demog}_n${nref}_t${ntgt}_s${seed}"

  echo "🚀 Submitting: $jobname"

  sbatch /home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/1src_sprime_run.sh "$demog" "$nref" "$ntgt" "$seed"

done < "$task_file"
