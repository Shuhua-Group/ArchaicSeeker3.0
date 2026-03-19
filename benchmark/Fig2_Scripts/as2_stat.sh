#!/bin/bash
#SBATCH --job-name=as2_stat
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --array=1-100
#SBATCH --mem=5G
#SBATCH --partition=batch
#SBATCH --exclude=cld002

source ~/.bashrc
conda activate Python

# seeds 文件
seeds="/home/linhuanyu/share1/20_AS3/0_Scripts/seeds.txt"
# 取出本任务对应的 seed（第 SLURM_ARRAY_TASK_ID 行）
seed=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" "$seeds")
# seed=36027470
cd /home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker2.0

for demo in AncientEurasia AS2_HumanNeanderthalDenisovan BonoboGhost ChimpBonoboGhost HumanArchaic HumanNeanderthal HumanNeanderthalDenisovan OOANeanderthal Skov_HumanDenisovan; do
    for nref in 10 50; do
        for ntgt in 1 10; do
            cd "/home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker2.0/${demo}/nref_${nref}/ntgt_${ntgt}/${seed}"
            python /home/linhuanyu/share1/20_AS3/0_Scripts/stats_AS2.py \
                --demo ${demo} \
                --nref ${nref} \
                --ntgt ${ntgt} \
                --seed ${seed}
        done
    done
done
