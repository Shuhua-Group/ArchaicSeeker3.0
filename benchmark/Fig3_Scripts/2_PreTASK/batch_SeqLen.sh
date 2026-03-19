#!/bin/bash
#SBATCH --job-name=Data_SeqLen
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
python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/3.1_SeqLen.py\
    --seed "$seed" \
    --maxLen 10

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/3.1_SeqLen.py\
    --seed "$seed" \
    --maxLen 20

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/3.1_SeqLen.py\
    --seed "$seed" \
    --maxLen 50

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/3.1_SeqLen.py\
    --seed "$seed" \
    --maxLen 75

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/3.1_SeqLen.py\
    --seed "$seed" \
    --maxLen 100

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/3.1_SeqLen.py\
    --seed "$seed" \
    --maxLen 150

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/3.1_SeqLen.py\
    --seed "$seed" \
    --maxLen 200



