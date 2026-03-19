#!/bin/bash
# 用法：
#   bash make_target_by_AC.sh Sim_VCF OUT_DIR
#
# 功能：
#   给定一个模拟 VCF：
#     1）提取所有 (CHR, POS)
#     2）在 0,10,...,90% 删除比例下，随机删除对应比例的位点（保持原顺序）
#     3）对每个删除比例，生成对应的：
#          - 目标样本 VCF：Sim.mask${p}.tgt.vcf.gz
#          - 参考样本 VCF：Sim.mask${p}.ref.vcf.gz
#        （ref/tgt 共享同一批位点）
#     4）记录每个比例下的 SNP 数量到 SNP_Density.txt

set -euo pipefail

module load bcftools/1.14

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 Sim_VCF OUT_DIR" >&2
    exit 1
fi

Sim_VCF="$1"
OUT_DIR="$2"

# 目标和参考样本列表
Tgtlist="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/7_SNP/tgt.list"
Reflist="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/7_SNP/ref.list"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

echo "[`date`] Sim_VCF : ${Sim_VCF}"
echo "[`date`] OUT_DIR  : ${OUT_DIR}"

# 如果原始 VCF 没有索引，顺手建一个
if [ ! -f "${Sim_VCF}.tbi" ] && [ ! -f "${Sim_VCF}.csi" ]; then
    echo "[`date`] Index original Sim_VCF ..."
    bcftools index -t "${Sim_VCF}"
fi

# 提取所有位点的 (CHR, POS)
if [ ! -f Sim.CHR.POS.txt ]; then
    echo "[`date`] Extract CHROM & POS from Sim_VCF ..."
    bcftools query -f "%CHROM\t%POS\n" "${Sim_VCF}" > Sim.CHR.POS.txt
fi

n_sites=$(wc -l < Sim.CHR.POS.txt || echo 0)
echo "[`date`] Total sites in Sim_VCF: ${n_sites}"

if [ "${n_sites}" -eq 0 ]; then
    echo "[`date`] No sites found in Sim_VCF. Exit."
    exit 1
fi

# 统计文件：记录每个 mask 比例下的 SNP 个数
echo -e "Label\tNumVariants" > SNP_Density.txt

# 先记录 full（不删除任何位点）的总 SNP 数
n_all=$(bcftools index -n "${Sim_VCF}")
echo -e "mask0\t${n_all}" >> SNP_Density.txt

# 要删除的比例（百分比）
mask_percents="0 10 20 30 40 50 60 70 80 90"

for p in ${mask_percents}; do
    echo "[`date`] Mask ${p}%"

    if [ "${p}" -eq 0 ]; then
        # 不删任何点，直接复制
        cp Sim.CHR.POS.txt Sim.mask${p}.sites.txt
    else
        # 用 awk 随机删除 p% 行（保留概率 = 1 - p/100），顺序天然不变
        awk -v p="$p" 'BEGIN{srand()} rand() > p/100' Sim.CHR.POS.txt \
            > Sim.mask${p}.sites.txt
    fi

    # 当前这个 mask 下实际保留的位点数
    n_keep=$(wc -l < Sim.mask${p}.sites.txt || echo 0)
    echo "[`date`]  kept ${n_keep}/${n_sites} sites"

    if [ "${n_keep}" -eq 0 ]; then
        echo "[`date`]  Mask ${p}%: 0 sites kept, skip VCF filtering."
        echo -e "mask${p}\t0" >> SNP_Density.txt
        continue
    fi

    # 生成目标样本 VCF（tgt）和参考样本 VCF（ref）
    tgt_vcf="target.mask${p}.vcf.gz"
    ref_vcf="ref.mask${p}.vcf.gz"

    echo "[`date`]  Generating ${tgt_vcf} and ${ref_vcf} ..."

    bcftools view \
      -S "${Tgtlist}" \
      -T Sim.mask${p}.sites.txt \
      -Oz \
      -o "${tgt_vcf}" \
      "${Sim_VCF}"

    bcftools index -t "${tgt_vcf}"

    bcftools view \
      -S "${Reflist}" \
      -T Sim.mask${p}.sites.txt \
      -Oz \
      -o "${ref_vcf}" \
      "${Sim_VCF}"

    bcftools index -t "${ref_vcf}"

    # 统计这个 mask 比例下的 SNP 数（tgt/ref 一样，取一个即可）
    n_var=$(bcftools index -n "${tgt_vcf}")
    echo -e "mask${p}\t${n_var}" >> SNP_Density.txt
done



rm -f all_lines.tmp

echo "[`date`] All done. SNP densities written to SNP_Density.txt"
