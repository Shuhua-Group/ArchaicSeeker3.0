#!/bin/bash

scripts=(
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/10.6_as3_run_ArchaicNum.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/10.7_as3_run_ArchaicNum.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/10.8_as3_run_ArchaicNum.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/10.9_as3_run_ArchaicNum.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/10.10_as3_run_ArchaicNum.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/10.11_as3_run_ArchaicNum.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/10.12_as3_run_ArchaicNum.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/10.13_as3_run_ArchaicNum.py"
)

for script in "${scripts[@]}"; do
    echo "正在运行: $script"
    python "$script" || echo "警告: $script 执行失败，继续下一个..."
    echo "--------------------------------------------------"
done

echo "所有任务执行完成"