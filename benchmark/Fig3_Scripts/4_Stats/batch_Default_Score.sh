#!/bin/bash
#SBATCH --job-name=Score
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --array=1-100
#SBATCH --mem=1G
#SBATCH --partition=batch

source ~/.bashrc
conda activate Python

# seeds 文件
seeds="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/0_Config/seeds.txt"
# 取出本任务对应的 seed（第 SLURM_ARRAY_TASK_ID 行）
seed=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" "$seeds")

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.2_Score.py \
    --seed "$seed" \
    --cutoff 0.3

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.2_Score.py \
    --seed "$seed" \
    --cutoff 0.4

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.2_Score.py \
    --seed "$seed" \
    --cutoff 0.5

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.2_Score.py \
    --seed "$seed" \
    --cutoff 0.55

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.2_Score.py \
    --seed "$seed" \
    --cutoff 0.6

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.2_Score.py \
    --seed "$seed" \
    --cutoff 0.65

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.2_Score.py \
    --seed "$seed" \
    --cutoff 0.7

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.2_Score.py \
    --seed "$seed" \
    --cutoff 0.75

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.2_Score.py \
    --seed "$seed" \
    --cutoff 0.8

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.2_Score.py \
    --seed "$seed" \
    --cutoff 0.85

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.2_Score.py \
    --seed "$seed" \
    --cutoff 0.9




