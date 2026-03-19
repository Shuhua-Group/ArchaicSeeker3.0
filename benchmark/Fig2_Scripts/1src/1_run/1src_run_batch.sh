#!/bin/bash
#SBATCH --job-name=run_batch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=1G
#SBATCH --partition=batch

# 任务列表
task_file="/home/linhuanyu/share1/20_AS3/0_Scripts/1src/0_batch/full_1src_list.txt"

# 切换到日志目录
cd /home/linhuanyu/share1/20_AS3/logs

# 逐行读取任务
while read -r demog nref ntgt seed; do
  jobname="${demog}_n${nref}_t${ntgt}_s${seed}"

  echo "🚀 Submitting: $jobname"

  sbatch /home/linhuanyu/share1/20_AS3/0_Scripts/1src/1_run/1src_run.sh "$demog" "$nref" "$ntgt" "$seed"

  # rel_path="${demog}/nref_${nref}/ntgt_${ntgt}/${seed}"
  # outdir="/home/linhuanyu/share1/20_AS3/results/inference/IBDmix/${rel_path}"
  # arc_list="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/IBDmix/${demog}/nref_${nref}/ntgt_${ntgt}/sim.arc.list"
  # modern_list="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/IBDmix/${demog}/nref_${nref}/ntgt_${ntgt}/sim.modern.list"
  # invcf="/home/linhuanyu/share1/20_AS3/results/simulated_data/${demog}/nref_${nref}/ntgt_${ntgt}/${seed}/sim1src.biallelic.vcf.gz"

  # sbatch --job-name="$jobname" /home/linhuanyu/share1/20_AS3/results/inference/0_Scripts/1src/1_run/ibdmix.sh "$outdir" "$arc_list" "$modern_list" "$invcf"
done < "$task_file"