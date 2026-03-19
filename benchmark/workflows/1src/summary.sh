#!/bin/bash
#SBATCH --job-name=summary
#SBATCH --output=summary_%j.out
#SBATCH --error=summary_%j.err
#SBATCH --array=0-999
#SBATCH --mem=1G
#SBATCH --partition=batch



source ~/.bashrc
conda activate sstar-analysis

for i in {0..1}; do
  chunk_file="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/task_1src_chunks/task_1src_chunk${i}.txt"
  if [[ -f "$chunk_file" ]]; then
    line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$chunk_file")
    if [[ -n "$line" ]]; then
      echo "🔹 Running chunk $i line $SLURM_ARRAY_TASK_ID: $line"
      # python /home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/2src/as3.0.py $line
      # python /home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/2src/as3.1.py $line
      python /home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/1src/ibdmix_stat.py $line
    else
      echo "⚠️  [chunk $i] line $SLURM_ARRAY_TASK_ID is empty"
    fi
  else
    echo "❌  [chunk $i] not found: $chunk_file"
  fi
done
