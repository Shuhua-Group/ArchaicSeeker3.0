#!/bin/bash
#SBATCH --job-name=Data_Default
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
python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.0_preMerge.py \
    --seed "$seed" \
    --prefix "AS3_Merge_0"

# python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.0_preMerge.py \
#     --seed "$seed" \
#     --prefix "AS3_Merge_2500"

# python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.0_preMerge.py \
#     --seed "$seed" \
#     --prefix "AS3_Merge_5000"

# python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.0_preMerge.py \
#     --seed "$seed" \
#     --prefix "AS3_Merge_7500"

# python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.0_preMerge.py \
#     --seed "$seed" \
#     --prefix "AS3_Merge_10000"

# python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.0_preMerge.py \
#     --seed "$seed" \
#     --prefix "AS3_Merge_12500"

# python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.0_preMerge.py \
#     --seed "$seed" \
#     --prefix "AS3_Merge_15000"

# python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.0_preMerge.py \
#     --seed "$seed" \
#     --prefix "AS3_Merge_17500"

# python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/1.0_preMerge.py \
#     --seed "$seed" \
#     --prefix "AS3_Merge_20000"


















