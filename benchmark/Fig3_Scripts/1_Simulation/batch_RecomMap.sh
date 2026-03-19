#!/bin/bash
#SBATCH --job-name=Simulation_RecomMap
#SBATCH --output=Simulation_RecomMap_%a.out
#SBATCH --error=Simulation_RecomMap_%a.err
#SBATCH --array=1-100
#SBATCH --mem=5G
#SBATCH --partition=batch

source ~/.bashrc
conda activate Python

# seeds 文件
seeds="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/0_Config/seeds.txt"

# 取出本任务对应的 seed（第 SLURM_ARRAY_TASK_ID 行）
seed=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" "$seeds")

base_outdir="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/2_WithMap"
outdir="${base_outdir}/${seed}"

mkdir -p "$outdir"
cd "$base_outdir"

python /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/1_Simulation/Simulation_RecomMap.py \
    --seed "$seed" \
    --outdir "$outdir"

