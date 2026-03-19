#!/usr/bin/env bash
#
# FUNCTION:   VCF preparation pipeline for archaic and modern human data.
#

set -euo pipefail

# ===================================================================================
# User Configuration
# NOTE: Modify the variables in this section to match your system's environment
#       and file locations.
# ===================================================================================

# --- Computational Resources ---
MAX_PROCS=1
THREADS_PER_JOB=6

# --- Input File Paths ---
REF_GENOME="examples/raw_data/reference_fa/GRCh38.22_demo.fa"
DENISOVAN_VCF_DIR="examples/raw_data/vcf"
NEANDERTHAL_VCF_DIR="examples/raw_data/vcf"
MODERN_HUMAN_VCF_DIR="examples/raw_data/vcf"
SAMPLE_LISTS_DIR="examples/raw_data/sample_info"
MASK_BED_DIR="examples/raw_data/high_quality_region_GRCh38"

# --- Input File Names ---
YRI_SAMPLES_FILE="YRI.sample.txt"
TARGET_SAMPLES_FILE="CHB.sample.txt"
REF_SAMPLES_FILE="ref.sample.txt"

# --- Output Directory ---
MAIN_OUTPUT_DIR="examples/raw_data_preprocessed_output"

# ===================================================================================

# Use `readonly CHROMOSOMES=($(seq 1 22))` for a full production run.
readonly CHROMOSOMES=(22) # Using a single chromosome for the example run.
readonly FINAL_REF_DIR="${MAIN_OUTPUT_DIR}/Final_Ref_VCFs"
readonly FINAL_TARGET_DIR="${MAIN_OUTPUT_DIR}/Final_Target_VCFs"
readonly YRI_LIST="${SAMPLE_LISTS_DIR}/${YRI_SAMPLES_FILE}"
readonly TARGET_LIST="${SAMPLE_LISTS_DIR}/${TARGET_SAMPLES_FILE}"
readonly REF_SAMPLES_LIST="${SAMPLE_LISTS_DIR}/${REF_SAMPLES_FILE}"

process_chromosome() {
    local -a MASK_PREFIXES=($MASK_PREFIXES_STR)
    local K=$1
    local start_time=$(date +%s)
    
    echo "[CHR ${K}] Processing started."

    local TMP_DIR="${MAIN_OUTPUT_DIR}/temp/chr${K}"
    mkdir -p "$TMP_DIR"
    trap 'rm -rf "${TMP_DIR}"' RETURN

    local den_in="${DENISOVAN_VCF_DIR}/Den.hg38.${K}.part.vcf.gz"
    local nea_in="${NEANDERTHAL_VCF_DIR}/Nean.hg38.${K}.part.vcf.gz"
    local modern_in="${MODERN_HUMAN_VCF_DIR}/KGP.GRCh38.PASS.snps.biallelic.shapeit4.${K}.part.vcf.gz"

    for f in "$den_in" "$nea_in" "$modern_in" "$YRI_LIST" "$TARGET_LIST" "$REF_SAMPLES_LIST"; do
        if [[ ! -f "$f" ]]; then
            echo "[CHR ${K}] ERROR: Required input file not found: $f" >&2
            return 1
        fi
    done

    echo "[CHR ${K}] Stage 1: Creating master site list."
    local modern_biallelic_norm="${TMP_DIR}/modern.biallelic.norm.vcf.gz"
    bcftools norm --threads "${THREADS_PER_JOB}" -f "$REF_GENOME" -c s "$modern_in" | \
        bcftools view -i 'N_ALT==1' -Oz -o "$modern_biallelic_norm"
    bcftools index -t "$modern_biallelic_norm"

    local target_sites_vcf="${TMP_DIR}/target.sites.vcf.gz"
    bcftools view --threads "${THREADS_PER_JOB}" -S "$TARGET_LIST" "$modern_biallelic_norm" | \
        bcftools view -i 'MAF>=0.001' -Oz -o "$target_sites_vcf"
    bcftools index -t "$target_sites_vcf"

    echo "[CHR ${K}] Stage 2: Preparing harmonized, GT-only VCFs."
    local den_ready_to_merge="${TMP_DIR}/den.ready.vcf.gz"
    bcftools view --threads "${THREADS_PER_JOB}" -T "$target_sites_vcf" "$den_in" | \
        bcftools norm --threads "${THREADS_PER_JOB}" -f "$REF_GENOME" -c s | \
        bcftools annotate -x INFO,^FORMAT/GT -Oz -o "$den_ready_to_merge"
    bcftools index -t "$den_ready_to_merge"

    local nea_ready_to_merge="${TMP_DIR}/nea.ready.vcf.gz"
    bcftools view --threads "${THREADS_PER_JOB}" -T "$target_sites_vcf" "$nea_in" | \
        bcftools norm --threads "${THREADS_PER_JOB}" -f "$REF_GENOME" -c s | \
        bcftools annotate -x INFO,^FORMAT/GT -Oz -o "$nea_ready_to_merge"
    bcftools index -t "$nea_ready_to_merge"

    local yri_ready_to_merge="${TMP_DIR}/yri.ready.vcf.gz"
    bcftools view --threads "${THREADS_PER_JOB}" -S "$YRI_LIST" "$modern_biallelic_norm" | \
        bcftools view -T "$target_sites_vcf" | \
        bcftools annotate -x INFO,^FORMAT/GT -Oz -o "$yri_ready_to_merge"
    bcftools index -t "$yri_ready_to_merge"

    echo "[CHR ${K}] Stage 3: Merging and filtering."
    local archaic_yri_merged="${TMP_DIR}/archaic_yri.merged.vcf.gz"
    bcftools merge --threads "${THREADS_PER_JOB}" --missing-to-ref -0 -Oz \
        -o "$archaic_yri_merged" "$den_ready_to_merge" "$nea_ready_to_merge" "$yri_ready_to_merge"
    bcftools index -t "$archaic_yri_merged"
        
    local isec_dir="${TMP_DIR}/isec"
    bcftools isec -p "$isec_dir" -n=2 -c none --threads "${THREADS_PER_JOB}" -Oz \
        "$target_sites_vcf" "$archaic_yri_merged"
    
    local final_merged="${TMP_DIR}/final.merged.vcf.gz"
    bcftools merge --threads "${THREADS_PER_JOB}" --force-samples -Oz \
        -o "$final_merged" "${isec_dir}/0000.vcf.gz" "${isec_dir}/0001.vcf.gz"
    
    local final_filtered="${TMP_DIR}/final.filtered.vcf.gz"
    bcftools view --threads "${THREADS_PER_JOB}" -i 'N_ALT > 0' \
        -Oz -o "$final_filtered" "$final_merged"
    bcftools index -t "$final_filtered"

    echo "[CHR ${K}] Stage 4: Applying high-quality region masks."
    local masked_vcf_input="$final_filtered"
    local final_masked_vcf=""

    for prefix in "${MASK_PREFIXES[@]}"; do
        local mask_bed_file="${MASK_BED_DIR}/${prefix}_chr${K}.bed"
        if [[ ! -f "$mask_bed_file" ]]; then
            echo "[CHR ${K}] ERROR: Mask file not found: ${mask_bed_file}" >&2
            return 1
        fi
        
        final_masked_vcf="${TMP_DIR}/final.masked.${prefix}.vcf.gz"
        echo "    -> Applying mask: ${mask_bed_file}"
        bcftools view --threads "${THREADS_PER_JOB}" -T "${mask_bed_file}" \
            -Oz -o "$final_masked_vcf" "$masked_vcf_input"
        bcftools index -t "$final_masked_vcf"
        
        masked_vcf_input="$final_masked_vcf"
    done
    
    if [[ -z "$final_masked_vcf" ]]; then
        echo "    -> INFO: No masks applied, proceeding with unmasked data."
        final_masked_vcf="$final_filtered" 
    fi

    echo "[CHR ${K}] Stage 5: Generating final split VCFs."
    local final_ref_vcf="${FINAL_REF_DIR}/ref_panel.chr${K}.vcf.gz"
    bcftools view --threads "${THREADS_PER_JOB}" -S "$REF_SAMPLES_LIST" --force-samples \
        -Oz -o "$final_ref_vcf" "$final_masked_vcf"
    bcftools index -t "$final_ref_vcf"
    
    local final_target_vcf="${FINAL_TARGET_DIR}/target_panel.chr${K}.vcf.gz"
    bcftools view --threads "${THREADS_PER_JOB}" -S "$TARGET_LIST" --force-samples \
        -Oz -o "$final_target_vcf" "$final_masked_vcf"
    bcftools index -t "$final_target_vcf"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "[CHR ${K}] Processing complete. Duration: ${duration} seconds."
}

MASK_PREFIXES=("N" "D")
MASK_PREFIXES_STR="${MASK_PREFIXES[*]}"
export MASK_PREFIXES_STR

export -f process_chromosome
export REF_GENOME DENISOVAN_VCF_DIR NEANDERTHAL_VCF_DIR MODERN_HUMAN_VCF_DIR \
       SAMPLE_LISTS_DIR YRI_LIST TARGET_LIST REF_SAMPLES_LIST \
       MAIN_OUTPUT_DIR FINAL_REF_DIR FINAL_TARGET_DIR THREADS_PER_JOB \
       MASK_BED_DIR

mkdir -p "$MAIN_OUTPUT_DIR/temp" "$FINAL_REF_DIR" "$FINAL_TARGET_DIR"

if [[ ! -f "$REF_GENOME" ]]; then echo "FATAL ERROR: Reference genome not found: $REF_GENOME"; exit 1; fi
if [[ ! -d "$SAMPLE_LISTS_DIR" ]]; then echo "FATAL ERROR: Sample lists directory not found: $SAMPLE_LISTS_DIR"; exit 1; fi
if [[ ! -d "$MASK_BED_DIR" ]]; then echo "FATAL ERROR: Mask BED directory not found: $MASK_BED_DIR"; exit 1; fi

echo "Starting VCF preparation pipeline..."

printf "%s\n" "${CHROMOSOMES[@]}" | xargs -n 1 -P "${MAX_PROCS}" -I {} bash -c "process_chromosome {}"

echo "All chromosome processing tasks are complete."

echo "Generating reference map file..."
REFERENCE_MAP_FILE="${MAIN_OUTPUT_DIR}/reference.map"

awk '{print $1 "\tAFR"}' "$YRI_LIST" > "$REFERENCE_MAP_FILE"

REPRESENTATIVE_CHR=${CHROMOSOMES[0]}
DEN_VCF_FOR_SAMPLES="${DENISOVAN_VCF_DIR}/Den.hg38.${REPRESENTATIVE_CHR}.part.vcf.gz"
NEA_VCF_FOR_SAMPLES="${NEANDERTHAL_VCF_DIR}/Nean.hg38.${REPRESENTATIVE_CHR}.part.vcf.gz"

if [[ -f "$DEN_VCF_FOR_SAMPLES" ]]; then
    bcftools query -l "$DEN_VCF_FOR_SAMPLES" | awk '{print $1 "\tDEN"}' >> "$REFERENCE_MAP_FILE"
else
    echo "WARNING: Could not find Denisovan VCF to extract samples: ${DEN_VCF_FOR_SAMPLES}"
fi

if [[ -f "$NEA_VCF_FOR_SAMPLES" ]]; then
    bcftools query -l "$NEA_VCF_FOR_SAMPLES" | awk '{print $1 "\tNEAN"}' >> "$REFERENCE_MAP_FILE"
else
    echo "WARNING: Could not find Neanderthal VCF to extract samples: ${NEA_VCF_FOR_SAMPLES}"
fi

echo "Reference map file created at: ${REFERENCE_MAP_FILE}"
echo "Pipeline finished."