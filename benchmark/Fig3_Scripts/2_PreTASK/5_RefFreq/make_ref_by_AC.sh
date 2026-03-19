#!/bin/bash
# 用法：
#   bash make_ref_by_AC.sh REF_VCF OUT_DIR
# 例如：
#   bash make_ref_by_AC.sh ref.vcf.gz /path/to/2_RefFreq

set -euo pipefail

module load bcftools/1.14

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 REF_VCF OUT_DIR" >&2
    exit 1
fi

# 路径参数
REF_LIST="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/1_Scripts/2_PreTASK/5_RefFreq/ref.list"
REF_VCF="$1"
OUT_DIR="$2"

if [ ! -f "${REF_LIST}" ]; then
    echo "[`date`] ERROR: REF_LIST not found: ${REF_LIST}" >&2
    exit 1
fi

if [ ! -f "${REF_VCF}" ]; then
    echo "[`date`] ERROR: REF_VCF not found: ${REF_VCF}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

echo "[`date`] REF_LIST: ${REF_LIST}"
echo "[`date`] REF_VCF : ${REF_VCF}"
echo "[`date`] OUT_DIR : ${OUT_DIR}"

echo "[`date`] Subset AFR samples from ref.vcf.gz ..."
bcftools view \
  -S "${REF_LIST}" \
  -Oz \
  -o AFR.vcf.gz \
  "${REF_VCF}"

bcftools index -t AFR.vcf.gz

echo "[`date`] Fill AC/AF/AN tags ..."
bcftools +fill-tags AFR.vcf.gz \
  -Oz \
  -o AFR.withAF.vcf.gz \
  -- -t AC,AF,AN

bcftools index -t AFR.withAF.vcf.gz

echo "[`date`] Export CHR POS AC ..."
bcftools query -f "%CHROM\t%POS\t%AC\n" AFR.withAF.vcf.gz > AFR.CHR.POS.AC.txt

echo "[`date`] Head of AFR.CHR.POS.AC.txt:"
head AFR.CHR.POS.AC.txt || true

echo "[`date`] Generate AC threshold site lists and filtered VCFs ..."
for t in 0 1 2 10 20; do
    awk -v T="$t" '$3 > T {print $1 "\t" $2}' AFR.CHR.POS.AC.txt > AC_gt${t}.txt
    sed -i '/^$/d' AC_gt${t}.txt

    n_sites=$(wc -l < AC_gt${t}.txt || echo 0)
    echo "[`date`]  AC > ${t}: ${n_sites} sites"

    if [ "${n_sites}" -eq 0 ]; then
        echo "[`date`]  AC > ${t}: no sites, skip VCF filtering."
        continue
    fi

    bcftools view \
      -T AC_gt${t}.txt \
      -Oz \
      -o ref.ACgt${t}.vcf.gz \
      "${REF_VCF}"

    bcftools index -t ref.ACgt${t}.vcf.gz
done

echo "[`date`] All done."
