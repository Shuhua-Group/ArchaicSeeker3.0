#!/bin/bash
# 用法：
#   bash make_target_by_AC.sh Target_VCF OUT_DIR

set -euo pipefail

module load bcftools/1.14

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 Target_VCF OUT_DIR" >&2
    exit 1
fi

Target_VCF="$1"
OUT_DIR="$2"

Tgtlist="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/5_RefFreq/tgt.list"

if [ ! -f "${Target_VCF}" ]; then
    echo "[`date`] ERROR: Target_VCF not found: ${Target_VCF}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

echo "[`date`] Target_VCF : ${Target_VCF}"
echo "[`date`] OUT_DIR    : ${OUT_DIR}"

# 如果你想，顺手给原始 VCF 建个 index（已存在的话不会报错）
if [ ! -f "${Target_VCF}.tbi" ] && [ ! -f "${Target_VCF}.csi" ]; then
    echo "[`date`] Index original Target_VCF ..."
    bcftools index -t "${Target_VCF}"
fi

echo "[`date`] Fill AC/AF/AN tags into Target_VCF ..."
bcftools +fill-tags "${Target_VCF}" \
  -Oz \
  -o Target.withAF.vcf.gz \
  -- -t AC,AF,AN

bcftools index -t Target.withAF.vcf.gz

echo "[`date`] Export CHR POS AC ..."
bcftools query -f "%CHROM\t%POS\t%AC\n" Target.withAF.vcf.gz > Target.CHR.POS.AC.txt

echo "[`date`] Head of Target.CHR.POS.AC.txt:"
head Target.CHR.POS.AC.txt || true

echo "[`date`] Generate AC threshold site lists and filtered VCFs ..."
thresholds="0 1 2 5 10 20 50 100 200 500 1000"

for t in ${thresholds}; do
    awk -v T="$t" '$3 > T {print $1 "\t" $2}' Target.CHR.POS.AC.txt > AC_gt${t}.txt
    sed -i '/^$/d' AC_gt${t}.txt

    n_sites=$(wc -l < AC_gt${t}.txt || echo 0)
    echo "[`date`]  AC > ${t}: ${n_sites} sites"

    if [ "${n_sites}" -eq 0 ]; then
        echo "[`date`]  AC > ${t}: no sites, skip VCF filtering."
        continue
    fi

    bcftools view \
      -S "${Tgtlist}" \
      -T AC_gt${t}.txt \
      -Oz \
      -o Target.ACgt${t}.vcf.gz \
      "${Target_VCF}"

    bcftools index -t Target.ACgt${t}.vcf.gz
done

echo "[`date`] Count variants (SNP density proxy) ..."
# 写一个带标签的统计输出，更直观
echo -e "Label\tNumVariants" > SNP_Density.txt
n_all=$(bcftools index -n Target.withAF.vcf.gz)
echo -e "AC_all\t${n_all}" >> SNP_Density.txt

for t in ${thresholds}; do
    vcf_file="Target.ACgt${t}.vcf.gz"
    if [ -f "${vcf_file}" ]; then
        n_var=$(bcftools index -n "${vcf_file}")
        echo -e "AC_gt${t}\t${n_var}" >> SNP_Density.txt
    else
        echo -e "AC_gt${t}\t0" >> SNP_Density.txt
    fi
done

echo "[`date`] All done."
