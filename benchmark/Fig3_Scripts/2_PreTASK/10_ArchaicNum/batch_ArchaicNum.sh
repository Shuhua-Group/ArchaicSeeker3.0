#!/bin/bash
#SBATCH --job-name=ArchaicNum
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --array=1-100
#SBATCH --mem=1G
#SBATCH --partition=batch
#SBATCH --exclude=cld002

source /home/linhuanyu/.bashrc
conda activate Python

# seeds 文件
seeds="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/0_Config/seeds.txt"

# 取出本任务对应的 seed（第 SLURM_ARRAY_TASK_ID 行）
seed=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" "$seeds")
python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/10_ArchaicNum/AnchaicNum.py \
    --seed "$seed"

