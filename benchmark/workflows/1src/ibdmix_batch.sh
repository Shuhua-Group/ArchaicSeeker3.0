#!/bin/bash
#SBATCH --job-name=ibdmix
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=1G
#SBATCH --partition=batch

task_file="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/task_1src_chunks/task_1src_chunk1.txt"
ibdmix_sh="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/1src/ibdmix.sh"
base_outdir="/home/linhuanyu/share1/20_AS3/results"
config_dir="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/IBDmix"
vcf_dir="${base_outdir}/simulated_data"
# rm -r /home/linhuanyu/00_log/05_AS3/2504/250426
# mkdir /home/linhuanyu/00_log/05_AS3/2504/250426
cd /home/linhuanyu/00_log/05_AS3/2506/250612

while read -r demog nref ntgt seed; do
  rel_path="${demog}/nref_${nref}/ntgt_${ntgt}/${seed}"
  outdir="${base_outdir}/inference/IBDmix/${rel_path}"
  arc_list="${config_dir}/${demog}/nref_${nref}/ntgt_${ntgt}/sim.arc.list"
  modern_list="${config_dir}/${demog}/nref_${nref}/ntgt_${ntgt}/sim.modern.list"
  invcf="${vcf_dir}/${rel_path}/sim1src.biallelic.vcf.gz"

  jobname="ibdmix_${demog}_n${nref}_t${ntgt}_s${seed}"

  echo "🚀 Submitting: $jobname"
  sbatch --job-name="$jobname" "$ibdmix_sh" "$outdir" "$arc_list" "$modern_list" "$invcf"
done < "$task_file"

# while read -r demog nref ntgt seed; do
#   rel_path="${demog}/nref_${nref}/ntgt_${ntgt}/${seed}"
#   outdir="${base_outdir}/inference/IBDmix/${rel_path}"
#   arc_list="${config_dir}/${demog}/nref_${nref}/ntgt_${ntgt}/sim.arc.list"
#   modern_list="${config_dir}/${demog}/nref_${nref}/ntgt_${ntgt}/sim.modern.list"
#   invcf="${vcf_dir}/${rel_path}/sim1src.biallelic.vcf.gz"

#   jobname="ibdmix_${demog}_n${nref}_t${ntgt}_s${seed}"

#   echo "🚀 Submitting: $jobname"
#   sbatch --job-name="$jobname" "$ibdmix_sh" "$outdir" "$arc_list" "$modern_list" "$invcf"
# done