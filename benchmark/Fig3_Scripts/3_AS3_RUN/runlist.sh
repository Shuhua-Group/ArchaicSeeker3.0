#!/bin/bash

scripts=(
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/3.1.7_as3_run_Len75Mb.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/3.1.6_as3_run_Len100Mb.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/8.1_as3_run_RefNum50.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/8.2_as3_run_RefNum100.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/8.3_as3_run_RefNum5.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/8.4_as3_run_RefNum10.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/8.5_as3_run_RefNum25.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/9.1_as3_run_TgtNum1.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/9.8_as3_run_TgtNum10.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/9.6_as3_run_TgtNum50.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/7.1_as3_run_Mask.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/7.2_as3_run_Mask10.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/7.3_as3_run_Mask20.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/7.4_as3_run_Mask30.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/7.6_as3_run_Mask50.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/7.7_as3_run_Mask60.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/9.1_as3_run_TgtNum1.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/9.8_as3_run_TgtNum10.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/9.7_as3_run_TgtNum25.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/9.5_as3_run_TgtNum100.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/9.4_as3_run_TgtNum250.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/9.3_as3_run_TgtNum500.py"
    "/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/3_AS3_RUN/9.2_as3_run_TgtNum1000.py"
)

for script in "${scripts[@]}"; do
    echo "正在运行: $script"
    python "$script" || echo "警告: $script 执行失败，继续下一个..."
    echo "--------------------------------------------------"
done

echo "所有任务执行完成"