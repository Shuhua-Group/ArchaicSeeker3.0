# 先写入第一个文件（包含第一行）
first_file=$(find /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut \
     -type f -name "gaps.summary.AS3_Merge_0.txt" | head -n 1)

cat "$first_file" > /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/PreMerge/gaps.summary.AS3_Merge_0.txt

# 拼接后续文件（跳过第一行）
find /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut \
     -type f -name "gaps.summary.AS3_Merge_0.txt" | tail -n +2 |
while read f; do
    tail -n +2 "$f" >> /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/PreMerge/gaps.summary.AS3_Merge_0.txt
done

# 先写入第一个文件（包含第一行）
first_file=$(find /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut \
     -type f -name "gaps_with_dist.AS3_Merge_0" | head -n 1)

cat "$first_file" > /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/PreMerge/gaps_with_dist.AS3_Merge_0.txt

# 拼接后续文件（跳过第一行）
find /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut \
     -type f -name "gaps_with_dist.AS3_Merge_0.txt" | tail -n +2 |
while read f; do
    tail -n +2 "$f" >> /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/1_Defalut/PreMerge/gaps_with_dist.AS3_Merge_0.txt
done
