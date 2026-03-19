#!/bin/bash
#SBATCH --job-name=2src_summary
#SBATCH --output=2src_summary_%j.out
#SBATCH --error=2src_summary_%j.err
#SBATCH --array=0-799
#SBATCH --mem=1G
#SBATCH --partition=batch



source ~/.bashrc
conda activate sstar-analysis

for i in {0..1}; do
  chunk_file="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/task_2src_chunks/task_2src_chunk${i}.txt"
  if [[ -f "$chunk_file" ]]; then
    line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$chunk_file")
    if [[ -n "$line" ]]; then
      echo "🔹 Running chunk $i line $SLURM_ARRAY_TASK_ID: $line"
      python /home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/as31_stat.py $line
      # python /home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/2src/as3.1.py $line
      # python /home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/2src/ibdmix_stat.py $line
    else
      echo "⚠️  [chunk $i] line $SLURM_ARRAY_TASK_ID is empty"
    fi
  else
    echo "❌  [chunk $i] not found: $chunk_file"
  fi
done
