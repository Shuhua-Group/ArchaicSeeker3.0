
# bcftools view /home/linhuanyu/share1/20_AS3/results/simulated_data/AncientEurasia/nref_10/ntgt_1/105983501/sim1src.biallelic.vcf.gz -O b -o sim1src.bcf
# bcftools index sim1src.bcf

# bcftools view \
#   -S /home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/SkovHMM/AncientEurasia/sim.ref10.tgt1.list \
#   -O b \
#   -o modern.bcf \
#   /home/linhuanyu/share1/20_AS3/results/simulated_data/AncientEurasia/nref_10/ntgt_1/105983501/sim1src.biallelic.vcf.gz
# bcftools index modern.bcf

# hmmix create_outgroup -ind=individuals.json -vcf=*.bcf -out=outgroup.txt
# hmmix mutation_rate -outgroup=outgroup.txt -out mutationrate.bed
# hmmix create_ingroup  -ind=individuals.json -vcf=*.bcf -out=obs -outgroup=outgroup.txt
# hmmix train  -obs=obs.tsk_10.txt -mutrates=mutationrate.bed -out=trained.tsk_10.json -haploid

# bcftools view \
#   -S /home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/SkovHMM/AncientEurasia/sim.arc.list \
#   -O b \
#   -o arc/arc1.bcf \
#   /home/linhuanyu/share1/20_AS3/results/simulated_data/AncientEurasia/nref_10/ntgt_1/105983501/sim1src.biallelic.vcf.gz
# bcftools index arc1.bcf


# hmmix decode -obs=obs.tsk_10.txt -mutrates=mutationrate.bed -param=trained.tsk_10.json -haploid -admixpop=arc1.bcf -out=tsk_10.decoded 
DEMOG="HumanArchaic"
NREF="10"
NTGT="1"
SEED="956714"

# # 配置路径（改成你的）
OUTPUT_DIR="/home/linhuanyu/share1/20_AS3/results/"
VCF_FILE="${OUTPUT_DIR}/simulated_data/${DEMOG}/nref_${NREF}/ntgt_${NTGT}/${SEED}/sim2src.biallelic.vcf.gz"

CONFIG_DIR="/home/linhuanyu/83_AS3_SSTAR/sstar-analysis/config/SkovHMM/${DEMOG}/nref_${NREF}/ntgt_${NTGT}"
INFER_DIR="${OUTPUT_DIR}/inference/SkovHMM/${DEMOG}/nref_${NREF}/ntgt_${NTGT}/${SEED}"

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

set -e  # 加这一句，脚本出错就退出

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

echo "🎉 All done!"


echo "✅ Done: $DEMOG nref=$NREF ntgt=$NTGT seed=$SEED"
