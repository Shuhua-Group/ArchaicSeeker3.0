#!/bin/bash
#SBATCH --job-name=run_batch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=1G
#SBATCH --partition=batch

# 任务列表
task_file="/home/linhuanyu/share1/20_AS3/0_Scripts/2src/0_batch/full_2src_list.txt"

# 切换到日志目录
cd /home/linhuanyu/share1/20_AS3/logs_IBDMix

# 逐行读取任务
while read -r demog nref ntgt seed; do
  jobname="${demog}_n${nref}_t${ntgt}_s${seed}"

  echo "🚀 Submitting: $jobname"

  sbatch /home/linhuanyu/share1/20_AS3/0_Scripts/2src/3_summary/combine.sh "$demog" "$nref" "$ntgt" "$seed"
done < "$task_file"