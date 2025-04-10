#!/bin/bash
#SBATCH --job-name=sirv_integration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=sirv_integration_%j.out
#SBATCH --error=sirv_integration_%j.err

# SIRV Integration Pipeline SLURM Script
# This script runs the SIRV Integration Pipeline on a SLURM cluster

# Define pipeline directory
PIPELINE_DIR="/data/gpfs/projects/punim2251/sirv-integration-pipeline"
SCRIPT_PATH="${PIPELINE_DIR}/run_sirv_pipeline.py"

# Set default values
RESTART_OPTION=""
FORCE_OPTION=""

# Process command line options
while getopts "rf" opt; do
    case $opt in
        r) RESTART_OPTION="--restart" ;;
        f) FORCE_OPTION="--force" ;;
        *) echo "Invalid option: -$OPTARG"; exit 1 ;;
    esac
done

# Change to pipeline directory and activate environment
cd $PIPELINE_DIR

# Activate Python environment if it exists
if [ -d "${PIPELINE_DIR}/sirv_env" ]; then
    echo "Activating Python environment..."
    source sirv_env/bin/activate
fi

# Set up paths to FLAMES data
FLAMES_DIR="/data/gpfs/projects/punim2251/Aim1_LongBench/ReadRarefaction_wFixedCells/data/LongBench_All/ont_sc/10Percent_FLAMES"
FLAMES_BAM="${FLAMES_DIR}/realign2transcript.bam"
FLAMES_GTF="${FLAMES_DIR}/isoform_annotated.gtf"

# Set up paths to SIRV data
SIRV_DATA_DIR="/data/gpfs/projects/punim2251/sirv_run_01_longbench"
SIRV_BAM="${SIRV_DATA_DIR}/sirv_alignment.bam"
SIRV_REF="${SIRV_DATA_DIR}/local_reference.fa"
SIRV_GTF="${SIRV_DATA_DIR}/auto_generated_reference.gtf"

# Default output directory
OUTPUT_DIR="${PIPELINE_DIR}/output"

# Check for required files
echo "Checking for required files..."

# Add any file existence checks here
if [ ! -f "${SIRV_BAM}" ]; then
    echo "ERROR: SIRV BAM file not found: ${SIRV_BAM}"
    exit 1
fi

if [ ! -f "${SIRV_REF}" ]; then
    echo "ERROR: SIRV reference file not found: ${SIRV_REF}"
    exit 1
fi

if [ ! -f "${SIRV_GTF}" ]; then
    echo "ERROR: SIRV GTF file not found: ${SIRV_GTF}"
    exit 1
fi

# Build the command based on available files
CMD="python ${SCRIPT_PATH} --integration --output-dir ${OUTPUT_DIR} --threads $SLURM_CPUS_PER_TASK --sirv-bam ${SIRV_BAM} --sirv-reference ${SIRV_REF} --sirv-gtf ${SIRV_GTF} --visualize-coverage --verbose"

# Add options for restart and force if specified
if [ -n "${RESTART_OPTION}" ]; then
    CMD="${CMD} ${RESTART_OPTION}"
fi

if [ -n "${FORCE_OPTION}" ]; then
    CMD="${CMD} ${FORCE_OPTION}"
fi

# Add FLAMES files if they exist
if [ -f "${FLAMES_BAM}" ]; then
    echo "Found FLAMES BAM file: ${FLAMES_BAM}"
    CMD="${CMD} --learn-coverage-from ${FLAMES_BAM} --coverage-model custom --min-reads 50 --length-bins 5"
    
    if [ -f "${FLAMES_GTF}" ]; then
        echo "Found FLAMES GTF file: ${FLAMES_GTF}"
        CMD="${CMD} --flames-gtf ${FLAMES_GTF}"
    else
        echo "FLAMES GTF file not found, will use SIRV GTF for modeling"
    fi
else
    echo "FLAMES BAM file not found, using default coverage model"
    CMD="${CMD} --coverage-model default"
fi

# Log and run the command
echo ""
echo "Running command: ${CMD}"
echo ""

# Execute the command
$CMD

# Check the exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline completed successfully"
else
    echo "Pipeline failed with exit code ${EXIT_CODE}"
    exit $EXIT_CODE
fi

# Deactivate virtual environment
deactivate

echo "Job completed at: $(date)" 