#!/bin/bash
#SBATCH --job-name=ibdmix
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=1G
#SBATCH --partition=batch

#!/bin/bash

task_file="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/task_2src_chunks/task_2src_chunk0.txt"
ibdmix_sh="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/2src/ibdmix.sh"
base_outdir="/home/linhuanyu/share1/20_AS3/results"
config_dir="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/IBDmix"
vcf_dir="${base_outdir}/simulated_data"

# 切换日志目录（可选）
cd /home/linhuanyu/00_log/05_AS3/2506/test

while read -r demog nref ntgt seed; do
  rel_path="${demog}/nref_${nref}/ntgt_${ntgt}/${seed}"
  outdir="${base_outdir}/inference/IBDmix/${rel_path}"

  arc1_list="${config_dir}/${demog}/nref_${nref}/ntgt_${ntgt}/sim.arc1.list"
  arc2_list="${config_dir}/${demog}/nref_${nref}/ntgt_${ntgt}/sim.arc2.list"
  modern_list="${config_dir}/${demog}/nref_${nref}/ntgt_${ntgt}/sim.modern.list"
  invcf="${vcf_dir}/${rel_path}/sim2src.biallelic.vcf.gz"

  jobname="ibdmix_${demog}_n${nref}_t${ntgt}_s${seed}"

  echo "🔍 Checking inputs for: $jobname"
  echo "  ➤ arc1: $arc1_list"
  echo "  ➤ arc2: $arc2_list"
  echo "  ➤ modern: $modern_list"
  echo "  ➤ invcf: $invcf"

  # 检查文件是否都存在
  if [[ ! -f "$arc1_list" ]]; then echo "❌ Missing: $arc1_list"; continue; fi
  if [[ ! -f "$arc2_list" ]]; then echo "❌ Missing: $arc2_list"; continue; fi
  if [[ ! -f "$modern_list" ]]; then echo "❌ Missing: $modern_list"; continue; fi
  if [[ ! -f "$invcf" ]]; then echo "❌ Missing: $invcf"; continue; fi

  echo "🚀 Submitting: $jobname"
  sbatch --job-name="$jobname" "$ibdmix_sh" "$outdir" "$arc1_list" "$arc2_list" "$modern_list" "$invcf"
done < "$task_file"



