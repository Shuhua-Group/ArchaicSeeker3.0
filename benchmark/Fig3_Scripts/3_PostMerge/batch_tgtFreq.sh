#!/bin/bash
#SBATCH --job-name=TgtFreq_FilterMerge
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --array=1-100
#SBATCH --mem=5G
#SBATCH --partition=batch
#SBATCH --exclude=cld002

source ~/.bashrc
conda activate Python

# seeds 文件
seeds="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/0_Config/seeds.txt"
# 取出本任务对应的 seed（第 SLURM_ARRAY_TASK_ID 行）
seed=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" "$seeds")
# cp introgression_prediction.raw.bed introgression_prediction.raw.bk.bed
# cp introgression_prediction.raw.snps.gz introgression_prediction.raw.bk.snps.gz
# cp introgression_prob_matrix.txt introgression_prob_matrix.bk.txt

for prefix in ACgt0 ACgt1 ACgt2 ACgt5 ACgt10 ACgt20 ACgt50 ACgt100 ACgt200 ACgt500 ACgt1000
do
    cd "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/6_TgtFreq/${seed}/${prefix}"
    cp introgression_prediction.raw.bed introgression_prediction.raw.bk.bed
    cp introgression_prediction.raw.snps.gz introgression_prediction.raw.bk.snps.gz
    cp introgression_prob_matrix.txt introgression_prob_matrix.bk.txt

    python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_PostMerge/FilterMerge_TgtFreq.py \
        --seed "$seed" \
        --prefix "$prefix"
done
