# 先写入第一个文件（包含第一行）
first_file=$(find /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut \
     -type f -name "siminfered.AS3_Merge_0.bed" | head -n 1)

cat "$first_file" > /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/Match/siminfered.AS3_Merge_0.bed

# 拼接后续文件（跳过第一行）
find /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut \
     -type f -name "siminfered.AS3_Merge_0.bed" | tail -n +2 |
while read f; do
    tail -n +2 "$f" >> /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/Match/siminfered.AS3_Merge_0.bed
done

first_file=$(find /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut \
     -type f -name "infersim.AS3_Merge_0.bed" | head -n 1)

cat "$first_file" > /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/Match/infersim.AS3_Merge_0.bed

find /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut \
     -type f -name "infersim.AS3_Merge_0.bed" | tail -n +2 |
while read f; do
    tail -n +2 "$f" >> /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/Match/infersim.AS3_Merge_0.bed
done
