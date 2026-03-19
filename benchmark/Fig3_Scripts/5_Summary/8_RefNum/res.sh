#!/bin/bash

base_dir="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/8_RefNum"

# 遍历所有 seed 目录，例如 30133327, 30133328, ...
for seed_dir in "${base_dir}"/*/; do
    echo "Processing seed dir: ${seed_dir}"

    cd "${seed_dir}" || continue

    # 遍历当前 seed 目录下的所有 maskX_resource_usage.tsv
    for ru in *_resource_usage.tsv; do
        # 如果不存在匹配文件（比如目录里没跑完），就跳过
        [ -e "$ru" ] || continue

        # mask 名，比如 mask0 / mask10 / mask20 ...
        mask=${ru%%_resource_usage.tsv}

        # ① resource usage 一行
        ru_line=$(<"${mask}_resource_usage.tsv")

        # ② accuracy 一行
        acc_line=$(<"${mask}.accuracy")

        # # ③ SNP density：从 SNP_Density.txt 中找 mask 对应的最后一行第二列
        # snp=$(awk -v m="$mask" '$1==m {val=$2} END{print val}' SNP_Density.txt)

        rm -f "${mask}.res"
        echo -e "${ru_line}\t${acc_line}" > "${mask}.res"
    done
done
