#!/bin/bash
#SBATCH --job-name=hmmix_chunk0
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --mem=5G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-999  # 替换XXX为任务数量减1，比如有20组任务就是0-19

# 加载环境
source ~/.bashrc
conda activate sstar-analysis  # 根据你的环境改

# 准备参数列表
TASK_LIST="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/workflows/new/task_1src_chunks/task_1src_chunk0.txt"  # ✨ 你需要提前准备好，每行一个参数组合

# 取当前任务参数
TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $TASK_LIST)
DEMOG=$(echo $TASK | cut -d' ' -f1)
NREF=$(echo $TASK | cut -d' ' -f2)
NTGT=$(echo $TASK | cut -d' ' -f3)
SEED=$(echo $TASK | cut -d' ' -f4)


OUTPUT_DIR="/home/linhuanyu/share1/20_AS3/results/"
VCF_FILE="${OUTPUT_DIR}/simulated_data/${DEMOG}/nref_${NREF}/ntgt_${NTGT}/${SEED}/sim1src.biallelic.vcf.gz"
CONFIG_DIR="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/SkovHMM/${DEMOG}/nref_${NREF}/ntgt_${NTGT}"
INFER_DIR="${OUTPUT_DIR}/inference/HMMix/${DEMOG}/nref_${NREF}/ntgt_${NTGT}/${SEED}"

mkdir -p "$INFER_DIR"
cd "$INFER_DIR"

bcftools view "$VCF_FILE" -O b -o sim.bcf
bcftools index sim.bcf

bcftools view \
  -S "${CONFIG_DIR}/sim.modern.list" \
  -O b \
  -o modern.bcf \
  "$VCF_FILE"
bcftools index modern.bcf

# Step 1: create_outgroup
hmmix create_outgroup \
    -ind="${CONFIG_DIR}/individuals.json" \
    -vcf=*.bcf \
    -out=outgroup.txt

# Step 2: mutation_rate
hmmix mutation_rate \
    -outgroup=outgroup.txt \
    -out=mutationrate.bed

# Step 3: create_ingroup
hmmix create_ingroup \
    -ind="${CONFIG_DIR}/individuals.json" \
    -vcf=*.bcf \
    -out=obs \
    -outgroup=outgroup.txt

mkdir -p arc

bcftools view \
  -S "${CONFIG_DIR}/sim.arc.list" \
  -O b \
  -o arc/arc.bcf \
  "$VCF_FILE"
bcftools index arc/arc.bcf

echo "Content of ${CONFIG_DIR}/sim.tgt.list:"
cat "${CONFIG_DIR}/sim.tgt.list"

while read TGT_SAMPLE || [ -n "$TGT_SAMPLE" ]; do  # 防止最后一行无换行符漏读
    echo "🔵 Processing sample: $TGT_SAMPLE"

    if [ ! -f "obs.${TGT_SAMPLE}.txt" ]; then
        echo "❌ Error: obs.${TGT_SAMPLE}.txt not found!"
        exit 1
    fi

    hmmix train \
        -obs="obs.${TGT_SAMPLE}.txt" \
        -mutrates=mutationrate.bed \
        -out="trained.${TGT_SAMPLE}.json" \
        -haploid

    hmmix decode \
        -obs="obs.${TGT_SAMPLE}.txt" \
        -mutrates=mutationrate.bed \
        -param="trained.${TGT_SAMPLE}.json" \
        -haploid \
        -admixpop=arc/arc.bcf \
        -out="${TGT_SAMPLE}.decoded"

    echo "✅ Finished $TGT_SAMPLE"

done < "${CONFIG_DIR}/sim.tgt.list"

echo "✅ Done: $DEMOG nref=$NREF ntgt=$NTGT seed=$SEED"