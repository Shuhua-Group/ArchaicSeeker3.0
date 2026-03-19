#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=5G
#SBATCH --partition=batch
#SBATCH --exclude=cld007

outdir=$1
arc_list=$2
modern_list=$3
invcf=$4
log="$outdir/ibdmix.log"

mkdir -p "$outdir"
cd $outdir

echo "[INFO] Starting IBDmix pipeline"

echo "[INFO] Extracting archaic samples from $arc_list"

if grep -q '[^[:space:]]' "$arc_list"; then
    sed -i 's/\r$//' "$arc_list"
    bcftools view -S "$arc_list" "$invcf" -o arc.vcf
else
    echo "[ERROR] Archaic sample list is empty or invalid!" >&2
    cat -A "$arc_list" >&2  # 打印内容用于 debug
    exit 1
fi


echo "[INFO] Extracting modern samples from $modern_list"

if [[ $(grep -cv '^$' "$modern_list") -eq 1 ]]; then
    sample=$(head -n 1 "$modern_list" | tr -d '\r')
    new_sample="${sample}_copy"

    echo "[INFO] Only one modern sample detected: $sample"
    echo "[INFO] Will create a copy named: $new_sample"

    temp_vcf="modern_single.vcf"
    final_vcf="modern.vcf"

    # 提取单个样本的 VCF
    bcftools view -s "$sample" "$invcf" -Ov -o "$temp_vcf"

    # 用 awk 复制最后一列作为新样本列
    awk -v new_sample="$new_sample" '
    BEGIN { OFS = "\t" }
    /^##/ { print; next }
    /^#CHROM/ {
        print $0, new_sample
        next
    }
    {
        print $0, $NF
    }' "$temp_vcf" > "$final_vcf"

    echo "[INFO] Created modern.vcf with two samples: $sample and $new_sample"

    rm -f "$temp_vcf"

else
    sed -i 's/\r$//' "$modern_list"
    bcftools view -S "$modern_list" "$invcf" -Ov -o modern.vcf
    echo "[INFO] modern.vcf extracted from multiple samples."
fi


echo "[INFO] Archaic and modern VCFs extracted"

/home/linhuanyu/02_Software/introgression/IBDmix/build/src/generate_gt \
    --archaic arc.vcf \
    --modern modern.vcf \
    --output gt.txt

echo "[INFO] GT file generated"

/home/linhuanyu/02_Software/introgression/IBDmix/build/src/ibdmix \
    --genotype gt.txt \
    --output ibdmix_output.txt

echo "[INFO] IBDmix finished"

