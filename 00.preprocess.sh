#!/usr/bin/env bash
# ===================================================================================
#
# FUNCTION:   AS3 input data preparation pipeline.
#
# STRATEGY:   1. Process modern human data first to generate a target site list
#                by filtering on Minor Allele Frequency (MAF).
#             2. Use the site list to efficiently filter archaic data, avoiding
#                a full, time-consuming normalization of archaic genomes.
#             3. Integrate all steps into a highly parallel, per-chromosome workflow.
#
# VERSION:    3.0 - Stable Release
#
# ===================================================================================

# Exit immediately if a command exits with a non-zero status, if an undefined
# variable is used, or if a command in a pipeline fails.
set -euo pipefail

# ===================================================================================
# --- USER CONFIGURATION ---
#
# IMPORTANT: Please modify the variables in this section to match your system's
#            environment and file locations.
# ===================================================================================

# --- Computational Resources ---

# Maximum number of chromosomes to process in parallel.
# For whole-genome analysis, this is typically 22.
MAX_PROCS=22

# Number of threads allocated to each bcftools job.
# Total threads used will be approximately MAX_PROCS * THREADS_PER_JOB.
THREADS_PER_JOB=6

# --- Input File Paths ---

# Full path to the reference genome FASTA file (e.g., T2T-CHM13v2.0).
REF_GENOME="/path/to/your/reference_genome.fa"

# Directory containing Denisovan VCF files, split by chromosome.
# The script expects files like: ${DENISOVAN_VCF_DIR}/output_se.chm13.chr${K}.vcf.gz
DENISOVAN_VCF_DIR="/path/to/your/denisovan_vcfs"

# Directory containing Neanderthal VCF files, split by chromosome.
# The script expects files like: ${NEANDERTHAL_VCF_DIR}/output_se.chm13.chr${K}.vcf.gz
NEANDERTHAL_VCF_DIR="/path/to/your/neanderthal_vcfs"

# Directory containing modern human VCF files (e.g., KGP, HPRC), split by chromosome.
# The script expects files like: ${MODERN_HUMAN_VCF_DIR}/CPC.HPRC.Phase1.CHM13v2.chr${K}.filtered.vcf.gz
MODERN_HUMAN_VCF_DIR="/path/to/your/modern_human_vcfs"

# Directory containing the sample list files.
SAMPLE_LISTS_DIR="/path/to/your/sample_lists"

# --- Input File Names ---

# Filename for the list of African (e.g., YRI) samples (one sample ID per line).
YRI_SAMPLES_FILE="afr_samples.txt"

# Filename for the list of target population samples.
TARGET_SAMPLES_FILE="target_samples.txt"

# Filename for the list of samples that will form the final reference panel (e.g., Archaic + YRI).
REF_SAMPLES_FILE="ref_samples.txt"

# --- Output Directory ---

# Main directory where all outputs and temporary files will be stored.
MAIN_OUTPUT_DIR="/path/to/your/analysis_output_directory"

# ===================================================================================
# --- END OF USER CONFIGURATION ---
# ===================================================================================

# --- Derived Variables ---
readonly CHROMOSOMES=($(seq 1 22))
readonly FINAL_REF_DIR="${MAIN_OUTPUT_DIR}/Final_Ref_VCFs"
readonly FINAL_TARGET_DIR="${MAIN_OUTPUT_DIR}/Final_Target_VCFs"
readonly YRI_LIST="${SAMPLE_LISTS_DIR}/${YRI_SAMPLES_FILE}"
readonly TARGET_LIST="${SAMPLE_LISTS_DIR}/${TARGET_SAMPLES_FILE}"
readonly REF_SAMPLES_LIST="${SAMPLE_LISTS_DIR}/${REF_SAMPLES_FILE}"

# --- Main Processing Function ---
process_chromosome() {
    local K=$1
    local start_time=$(date +%s)
    
    echo "🚀 [CHR ${K}] Processing started. PID: $$"

    local TMP_DIR="${MAIN_OUTPUT_DIR}/temp/chr${K}"
    mkdir -p "$TMP_DIR"
    # Ensure temporary directory is cleaned up upon function exit
    trap 'echo "🧹 [CHR ${K}] Cleaning up temporary directory: ${TMP_DIR}"; rm -rf "${TMP_DIR}";' RETURN

    # Construct full paths to input files for this chromosome
    local den_in="${DENISOVAN_VCF_DIR}/output_se.chm13.chr${K}.vcf.gz"
    local nea_in="${NEANDERTHAL_VCF_DIR}/output_se.chm13.chr${K}.vcf.gz"
    local kgp_in="${MODERN_HUMAN_VCF_DIR}/CPC.HPRC.Phase1.CHM13v2.chr${K}.filtered.vcf.gz"

    # Validate that all required input files exist before starting
    for f in "$den_in" "$nea_in" "$kgp_in" "$YRI_LIST" "$TARGET_LIST" "$REF_SAMPLES_LIST"; do
        if [[ ! -f "$f" ]]; then
            echo "❌ [CHR ${K}] ERROR: Required input file not found: $f" >&2
            return 1
        fi
    done

    echo "[CHR ${K}] Stage 1: Processing modern human data..."
    local kgp_norm="${TMP_DIR}/kgp.norm.vcf.gz"
    bcftools norm --threads "${THREADS_PER_JOB}" -f "$REF_GENOME" -c s -Oz -o "$kgp_norm" "$kgp_in"
    
    local kgp_target_norm="${TMP_DIR}/kgp.target.norm.vcf.gz"
    bcftools view --threads "${THREADS_PER_JOB}" -S "$TARGET_LIST" -Oz -o "$kgp_target_norm" "$kgp_norm"
    
    local kgp_target_filtered="${TMP_DIR}/kgp.target.filtered.sites.vcf.gz"
    echo "     -> Filtering target population by MAF>=0.001 (critical step)..."
    bcftools view --threads "${THREADS_PER_JOB}" -i 'MAF>=0.001' -Oz -o "$kgp_target_filtered" "$kgp_target_norm"
    bcftools index -t "$kgp_target_filtered"
    
    local kgp_yri_norm="${TMP_DIR}/kgp.yri.norm.vcf.gz"
    echo "     -> Extracting YRI samples..."
    bcftools view --threads "${THREADS_PER_JOB}" -S "$YRI_LIST" -Oz -o "$kgp_yri_norm" "$kgp_norm"
    bcftools index -t "$kgp_yri_norm"

    echo "[CHR ${K}] Stage 2: Efficiently filtering and normalizing archaic data..."
    local den_norm_subset="${TMP_DIR}/denisova.norm.subset.vcf.gz"
    local nea_norm_subset="${TMP_DIR}/neanderthal.norm.subset.vcf.gz"
    
    echo "     -> Filtering & normalizing Denisovan..."
    bcftools view --threads "${THREADS_PER_JOB}" -T "$kgp_target_filtered" "$den_in" | \
    bcftools norm --threads "${THREADS_PER_JOB}" -f "$REF_GENOME" -c s -Oz -o "$den_norm_subset"
    bcftools index -t "$den_norm_subset"

    echo "     -> Filtering & normalizing Neanderthal..."
    bcftools view --threads "${THREADS_PER_JOB}" -T "$kgp_target_filtered" "$nea_in" | \
    bcftools norm --threads "${THREADS_PER_JOB}" -f "$REF_GENOME" -c s -Oz -o "$nea_norm_subset"
    bcftools index -t "$nea_norm_subset"

    echo "[CHR ${K}] Stage 3: Merging, intersecting, and final filtering..."
    
    local archaic_yri_merged="${TMP_DIR}/Archaic.YRI.merged.vcf.gz"
    echo "     -> Merging Archaic + YRI..."
    bcftools merge --threads "${THREADS_PER_JOB}" --missing-to-ref -0 -Oz \
        -o "$archaic_yri_merged" "$den_norm_subset" "$nea_norm_subset" "$kgp_yri_norm"
    bcftools index -t "$archaic_yri_merged"
        
    local isec_dir="${TMP_DIR}/isec"
    echo "     -> Intersecting sites between target population and (Archaic+YRI)..."
    bcftools isec -p "$isec_dir" -n=2 -c none --threads "${THREADS_PER_JOB}" -Oz \
        "$kgp_target_filtered" "$archaic_yri_merged"
    
    local final_merged="${TMP_DIR}/final.merged.vcf.gz"
    echo "     -> Merging intersection results..."
    bcftools merge --threads "${THREADS_PER_JOB}" --force-samples -Oz \
        -o "$final_merged" "${isec_dir}/0000.vcf.gz" "${isec_dir}/0001.vcf.gz"
    
    local final_filtered="${TMP_DIR}/final.filtered.vcf.gz"
    echo "     -> Filtering non-variant sites using 'N_ALT > 0'..."
    bcftools view --threads "${THREADS_PER_JOB}" -i 'N_ALT > 0' \
        -Oz -o "$final_filtered" "$final_merged"
    bcftools index -t "$final_filtered"

    echo "[CHR ${K}] Stage 4: Generating final split VCF files..."
    
    local final_ref_vcf="${FINAL_REF_DIR}/Archaic.AFR.ref.chr${K}.vcf.gz"
    bcftools view --threads "${THREADS_PER_JOB}" -S "$REF_SAMPLES_LIST" --force-samples \
        -Oz -o "$final_ref_vcf" "$final_filtered"
    bcftools index -t "$final_ref_vcf"
    
    local final_target_vcf="${FINAL_TARGET_DIR}/HGDPTGP_target.chr${K}.vcf.gz"
    bcftools view --threads "${THREADS_PER_JOB}" -S "$TARGET_LIST" --force-samples \
        -Oz -o "$final_target_vcf" "$final_filtered"
    bcftools index -t "$final_target_vcf"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "[CHR ${K}] Processing complete! Duration: ${duration} seconds. Final files are in ${FINAL_REF_DIR} and ${FINAL_TARGET_DIR}"
}

export -f process_chromosome
export REF_GENOME DENISOVAN_VCF_DIR NEANDERTHAL_VCF_DIR MODERN_HUMAN_VCF_DIR \
       SAMPLE_LISTS_DIR YRI_LIST TARGET_LIST REF_SAMPLES_LIST \
       MAIN_OUTPUT_DIR FINAL_REF_DIR FINAL_TARGET_DIR THREADS_PER_JOB

mkdir -p "$MAIN_OUTPUT_DIR/temp" "$FINAL_REF_DIR" "$FINAL_TARGET_DIR"

if [[ ! -f "$REF_GENOME" ]]; then
    echo "FATAL ERROR: Reference genome file not found at: $REF_GENOME"
    exit 1
fi
if [[ ! -d "$SAMPLE_LISTS_DIR" ]]; then
    echo "FATAL ERROR: Directory for sample lists not found at: $SAMPLE_LISTS_DIR"
    exit 1
fi

echo "==========================================================="
echo "Max Parallel Jobs: ${MAX_PROCS}"
echo "Threads per Job:   ${THREADS_PER_JOB}"
echo "Main Output Dir:   ${MAIN_OUTPUT_DIR}"
echo "==========================================================="

# Use xargs to run the process_chromosome function in parallel for each chromosome
printf "%s\n" "${CHROMOSOMES[@]}" | xargs -n 1 -P "${MAX_PROCS}" -I {} bash -c "process_chromosome {}"

echo "========================================"
echo "All chromosome processing tasks are complete!"
echo "========================================"