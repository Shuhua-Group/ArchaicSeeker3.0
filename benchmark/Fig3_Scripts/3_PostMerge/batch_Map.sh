#!/bin/bash
#SBATCH --job-name=0_FilterMerge
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --array=1-100
#SBATCH --mem=5G
#SBATCH --partition=batch

source ~/.bashrc
conda activate Python

# seeds 文件
seeds="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/0_Config/seeds.txt"
# 取出本任务对应的 seed（第 SLURM_ARRAY_TASK_ID 行）
seed=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" "$seeds")
cd "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut/${seed}"
# cp introgression_prediction.raw.bed introgression_prediction.raw.bk.bed
# cp introgression_prediction.raw.snps.gz introgression_prediction.raw.bk.snps.gz
# cp introgression_prob_matrix.txt introgression_prob_matrix.bk.txt


# for min_length in 0 0.5 1 1.5 1.7 2 2.5 3 4 5
# do
#     for min_score in 0 0.4 0.5 0.55 0.6 0.65 0.7 0.75 0.8
#     do
#         python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_PostMerge/filterMerge.py \
#             --seed "$seed" \
#             --min_length 5 \
#             --min_score 0.5 \
#             --distance 5000
#     done
# done

for distance in 4000 6000 7000 8000 9000 11000 12000
do
    python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_PostMerge/filterMerge.py \
        --seed "$seed" \
        --min_length 3 \
        --min_score 0.65 \
        --distance "$distance"
    
    python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_PostMerge/filterMerge.py \
        --seed "$seed" \
        --min_length 5 \
        --min_score 0 \
        --distance "$distance"
done