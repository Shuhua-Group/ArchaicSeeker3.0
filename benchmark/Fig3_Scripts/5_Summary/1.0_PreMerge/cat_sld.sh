find /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut \
    -type f -name "temp_5.0kb_s0.0_d*.accuracy" \
    -exec cat {} + \
    > /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/ScoreLengthMerge/Temp_5.0kb_s0.0.accuracy


find /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut \
    -type f -name "temp_3.0kb_s0.65_d*.accuracy" \
    -exec cat {} + \
    > /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/ScoreLengthMerge/Temp_3.0kb_s0.65.accuracy