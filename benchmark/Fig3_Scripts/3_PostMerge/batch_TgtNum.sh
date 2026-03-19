#!/bin/bash
#SBATCH --job-name=TgtNum_FilterMerge
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --array=1-100
#SBATCH --mem=20G
#SBATCH --partition=batch
#SBATCH --exclude=cld002

source ~/.bashrc
conda activate Python

# seeds 文件
seeds="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/0_Config/seeds.txt"
# 取出本任务对应的 seed（第 SLURM_ARRAY_TASK_ID 行）
seed=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" "$seeds")
cd "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/9_TgtNum/${seed}"
# cp introgression_prediction.raw.bed introgression_prediction.raw.bk.bed
# cp introgression_prediction.raw.snps.gz introgression_prediction.raw.bk.snps.gz
# cp introgression_prob_matrix.txt introgression_prob_matrix.bk.txt

for tgt_num in 1 10 25 50 100 250 500 1000
do
    cd "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/9_TgtNum/${seed}/${tgt_num}"
    cp introgression_prediction.raw.bed introgression_prediction.raw.bk.bed
    cp introgression_prediction.raw.snps.gz introgression_prediction.raw.bk.snps.gz
    cp introgression_prob_matrix.txt introgression_prob_matrix.bk.txt

    python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_PostMerge/TgtNum_FilterMerge.py \
        --seed "$seed" \
        --tgt_num "$tgt_num"
done
