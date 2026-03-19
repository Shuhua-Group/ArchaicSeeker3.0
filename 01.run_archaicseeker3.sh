#!/usr/bin/env bash
#
# FUNCTION: Run ArchaicSeeker3 analysis in parallel across multiple GPUs.

set -euo pipefail

# ==============================================================================
# --- CONFIGURATION ---
# Please modify the variables in this section before running the script.
# ==============================================================================

# 1. Path to the ArchaicSeeker3 main python script.
ASEEK_EXEC="ArchaicSeeker3.1-mamba"

# 2. Path to the main output directory from the data pre-processing step.
#    This directory will be used as the INPUT for this analysis script.
# DATA_IN_PATH="examples/preprocessed"
DATA_IN_PATH="examples/raw_data_preprocessed_output"

# 3. Root directory for the results of this ArchaicSeeker3 analysis.
ANALYSIS_OUT_ROOT="examples/aseeker3_results_$(date +%Y%m%d)"

# 4. GPU Configuration: list the IDs of the GPUs you want to use.
#GPUS=(0 1 2 3)
GPUS=(0)

readonly NGPUS=${#GPUS[@]}
readonly TARGET_VCF_DIR="${DATA_IN_PATH}/Final_Target_VCFs"
readonly REF_VCF_DIR="${DATA_IN_PATH}/Final_Ref_VCFs"
readonly MAP_FILE="${DATA_IN_PATH}/reference.map"

if ! { [ -f "$ASEEK_EXEC" ] && [ -d "$TARGET_VCF_DIR" ] && [ -d "$REF_VCF_DIR" ] && [ -f "$MAP_FILE" ]; }; then
    echo "ERROR: Missing required inputs. Please check the following paths in the CONFIGURATION section:"
    echo "   - ArchaicSeeker3 Executable: ${ASEEK_EXEC}"
    echo "   - Target VCF Directory:    ${TARGET_VCF_DIR}"
    echo "   - Reference VCF Directory: ${REF_VCF_DIR}"
    echo "   - Map File:                ${MAP_FILE}"
    exit 1
fi

echo "INFO: Starting ArchaicSeeker3 analysis..."
echo "INFO: Results will be saved to: ${ANALYSIS_OUT_ROOT}"
mkdir -p "${ANALYSIS_OUT_ROOT}"

for GPU in "${GPUS[@]}"; do
  (
    echo "INFO: Spawning worker for GPU ${GPU}..."
    # for K in {1..22}; do
    for K in 22; do
      if (( (K-1) % NGPUS == GPU )); then
        
        # Define input and output paths for the current chromosome
        target_vcf="${TARGET_VCF_DIR}/target_panel.chr${K}.vcf.gz"
        ref_vcf="${REF_VCF_DIR}/ref_panel.chr${K}.vcf.gz"
        outdir="${ANALYSIS_OUT_ROOT}/chr${K}"

        # Skip if VCF files for the current chromosome do not exist
        if [[ ! -f "$target_vcf" ]] || [[ ! -f "$ref_vcf" ]]; then
            echo "WARN [GPU ${GPU}]: VCF files for chr${K} not found, skipping."
            continue
        fi

        mkdir -p "${outdir}"
        echo "  [GPU ${GPU}] ==> Processing chr${K}..."
        
        # Run ArchaicSeeker3
        CUDA_VISIBLE_DEVICES=${GPU} python3 "${ASEEK_EXEC}" \
          -t "${target_vcf}" \
          -r "${ref_vcf}" \
          -m "${MAP_FILE}" \
          --merge 5000 \
          -o "${outdir}"
          
        echo "  [GPU ${GPU}] <== chr${K} processing finished."
      fi
    done
    echo "INFO: Worker for GPU ${GPU} has completed all assigned tasks."
  ) &
done

wait
echo "All jobs completed. ArchaicSeeker3 analysis has finished for all chromosomes."
