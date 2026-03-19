#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=5G
#SBATCH --partition=batch
#SBATCH --exclude=cld007

outdir=$1
arc1_list=$2
arc2_list=$3
modern_list=$4
invcf=$5
log="$outdir/ibdmix.log"

mkdir -p "$outdir"
cd "$outdir" || exit 1

exec &> >(tee -a "$log")

echo "[INFO] Starting IBDmix 2src pipeline"

# === 函数：检查 list 文件是否存在、非空、无 \r ===
check_list_file() {
    local label=$1
    local list_path=$2

    if [[ ! -f "$list_path" ]]; then
        echo "[ERROR] $label sample list does not exist: $list_path" >&2
        exit 1
    fi

    if [[ ! -s "$list_path" ]]; then
        echo "[ERROR] $label sample list is empty: $list_path" >&2
        exit 1
    fi

    # 删除 Windows \r 符号
    sed -i 's/\r$//' "$list_path"
    echo "[INFO] Valid $label list found: $list_path"
}

# === 检查并提取 archaic1 ===
check_list_file "Archaic1" "$arc1_list"
bcftools view -S "$arc1_list" "$invcf" -o arc1.vcf
echo "[INFO] arc1.vcf generated"

# === 检查并提取 archaic2 ===
check_list_file "Archaic2" "$arc2_list"
bcftools view -S "$arc2_list" "$invcf" -o arc2.vcf
echo "[INFO] arc2.vcf generated"

# === 提取 modern 样本 ===
check_list_file "Modern" "$modern_list"
echo "[INFO] Extracting modern samples from $modern_list"

if [[ $(grep -cv '^$' "$modern_list") -eq 1 ]]; then
    sample=$(head -n 1 "$modern_list" | tr -d '\r')
    new_sample="${sample}_copy"

    echo "[INFO] Only one modern sample detected: $sample"
    echo "[INFO] Will create a copy named: $new_sample"

    temp_vcf="modern_single.vcf"
    final_vcf="modern.vcf"

    bcftools view -s "$sample" "$invcf" -Ov -o "$temp_vcf"

    # 用 awk 复制 genotype 列
    awk -v new_sample="$new_sample" '
    BEGIN { OFS = "\t" }
    /^##/ { print; next }
    /^#CHROM/ { print $0, new_sample; next }
    { print $0, $NF }' "$temp_vcf" > "$final_vcf"

    echo "[INFO] Created modern.vcf with two samples: $sample and $new_sample"
    rm -f "$temp_vcf"
else
    bcftools view -S "$modern_list" "$invcf" -Ov -o modern.vcf
    echo "[INFO] modern.vcf extracted from multiple samples."
fi

# === IBDmix Part 1 ===
echo "[INFO] Running arc1 set"
/home/linhuanyu/02_Software/introgression/IBDmix/build/src/generate_gt \
    --archaic arc1.vcf \
    --modern modern.vcf \
    --output gt_1.txt
echo "[INFO] GT1 generated"

 /home/linhuanyu/02_Software/introgression/IBDmix/build/src/ibdmix \
    --genotype gt_1.txt \
    --output ibdmix_arc1_output.txt
echo "[INFO] ibdmix_arc1_output.txt done"

# === IBDmix Part 2 ===
echo "[INFO] Running arc2 set"
/home/linhuanyu/02_Software/introgression/IBDmix/build/src/generate_gt \
    --archaic arc2.vcf \
    --modern modern.vcf \
    --output gt_2.txt
echo "[INFO] GT2 generated"

 /home/linhuanyu/02_Software/introgression/IBDmix/build/src/ibdmix \
    --genotype gt_2.txt \
    --output ibdmix_arc2_output.txt
echo "[INFO] ibdmix_arc2_output.txt done"

echo "[INFO] IBDmix 2src finished."
