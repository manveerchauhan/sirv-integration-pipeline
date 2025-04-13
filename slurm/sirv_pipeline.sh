#!/bin/bash
#SBATCH --job-name=SIRV_Pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=physical
#SBATCH --output=sirv_pipeline_%j.out
#SBATCH --error=sirv_pipeline_%j.err

# SIRV Integration Pipeline SLURM Script
# --------------------------------------
# This script runs the SIRV Integration Pipeline in a SLURM environment.
# It uses the Python module installed via pip or conda.

# Configuration - modify these parameters
# --------------------------------------
OUTPUT_DIR="./sirv_run_$(date +%Y%m%d_%H%M%S)"  # Output directory
SIRV_REFERENCE="reference/SIRV_isoforms_C_170612a.fasta"  # SIRV reference FASTA
SIRV_GTF="reference/SIRV_isoforms_C_170612a.gtf"          # SIRV annotation GTF
SIRV_BAM="data/sirv_reads.bam"                   # BAM file with SIRV reads
SC_FASTQ="data/sc_reads.fastq"                   # FASTQ file with scRNA-seq reads
THREADS=8                                         # Number of threads to use
INSERTION_RATE=0.1                                # Target SIRV:scRNA-seq ratio
COVERAGE_MODEL="random_forest"                    # Coverage bias model type (only random_forest supported)
VISUALIZATION=true                                # Generate visualizations
COMPARATIVE=true                                  # Run comparative analysis
EXTRACT_FEATURES=true                             # Extract sequence features for random forest model

# Feature cache file for faster re-training
FEATURE_CACHE_FILE="${OUTPUT_DIR}/feature_cache.pkl"

# --------------------------------------
# DO NOT MODIFY BELOW THIS LINE
# --------------------------------------

echo "=== SIRV Integration Pipeline ==="
echo "Starting at $(date)"
echo "Output directory: ${OUTPUT_DIR}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log environment information
echo "=== Environment Information ===" > "${OUTPUT_DIR}/environment.log"
hostname >> "${OUTPUT_DIR}/environment.log"
echo "Date: $(date)" >> "${OUTPUT_DIR}/environment.log"
echo "SLURM Job ID: ${SLURM_JOB_ID}" >> "${OUTPUT_DIR}/environment.log"
echo "Python version: $(python --version 2>&1)" >> "${OUTPUT_DIR}/environment.log"
pip freeze >> "${OUTPUT_DIR}/environment.log"

# Prepare command
CMD="python -m sirv_pipeline --output-dir ${OUTPUT_DIR}"
CMD="${CMD} --integration"
CMD="${CMD} --sirv-reference ${SIRV_REFERENCE}"
CMD="${CMD} --sirv-gtf ${SIRV_GTF}"
CMD="${CMD} --sirv-bam ${SIRV_BAM}"
CMD="${CMD} --sc-fastq ${SC_FASTQ}"
CMD="${CMD} --integration-rate ${INSERTION_RATE}"
CMD="${CMD} --coverage-model ${COVERAGE_MODEL}"
CMD="${CMD} --threads ${THREADS}"

# Add feature cache if enabled
if [[ ! -z "${FEATURE_CACHE_FILE}" ]]; then
    CMD="${CMD} --feature-cache-file ${FEATURE_CACHE_FILE}"
fi

# Add optional flags
if [[ "${VISUALIZATION}" == "true" ]]; then
    CMD="${CMD} --visualize-coverage"
fi

if [[ "${COMPARATIVE}" == "true" ]]; then
    CMD="${CMD} --run-comparative-analysis"
fi

if [[ "${EXTRACT_FEATURES}" == "true" ]]; then
    CMD="${CMD} --extract-features"
fi

# Run the pipeline
echo "Command: ${CMD}"
echo "Command: ${CMD}" > "${OUTPUT_DIR}/command.txt"

# Execute command
eval "${CMD}"

RETVAL=$?

# Log completion
echo "Pipeline completed with exit code ${RETVAL} at $(date)" > "${OUTPUT_DIR}/completion.log"

# Exit with the pipeline's return code
exit ${RETVAL} 