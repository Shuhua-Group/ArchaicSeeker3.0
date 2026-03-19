#!/bin/bash
#SBATCH --job-name=Stat
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

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/0_StdStat.py \
    --seed "$seed" \
    --prefix "AS3_Merge_5000"

# for distance in 0 1000 2000 3000 4000 6000 7000 8000 9000 11000 12000
# do
#     python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/0_StdStat.py \
#         --seed "$seed" \
#         --prefix "temp_5.0kb_s0.0_d${distance}"
    
#     python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/4_Stats/0_StdStat.py \
#         --seed "$seed" \
#         --prefix "temp_3.0kb_s0.65_d${distance}"
# done