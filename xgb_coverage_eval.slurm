#!/bin/bash
#SBATCH --partition="sapphire"
#SBATCH --nodes=1
#SBATCH --account="punim2251"
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mschauhan@student.unimelb.edu.au
#SBATCH --cpus-per-task=12
#SBATCH --mem=100gb
#SBATCH --time=0-6:00:00
#SBATCH --job-name="xgb_coverage_eval"

# Create timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Define pipeline directory
PIPELINE_DIR="/data/gpfs/projects/punim2251/sirv-integration-pipeline"

# Check if a directory was provided as an argument
if [ $# -eq 1 ]; then
    # Use the provided directory
    BASE_OUTPUT_DIR="$1"
    echo "Using provided output directory: ${BASE_OUTPUT_DIR}"
else
    # Create a new timestamped directory
    BASE_OUTPUT_DIR="/data/gpfs/projects/punim2251/xgb_coverage_eval_${TIMESTAMP}"
    echo "Creating new output directory: ${BASE_OUTPUT_DIR}"
fi

echo "Creating output directory: ${BASE_OUTPUT_DIR}"

# Create base output directory if it doesn't exist
mkdir -p ${BASE_OUTPUT_DIR}

# Define input files and parameters
FLAMES_BAM="/data/gpfs/projects/punim2251/Aim1_LongBench/ReadRarefaction_wFixedCells/data/LongBench_All/ont_sc/10Percent_FLAMES/realign2transcript.bam"
FLAMES_GTF="/data/gpfs/projects/punim2251/Aim1_LongBench/ReadRarefaction_wFixedCells/data/LongBench_All/ont_sc/10Percent_FLAMES/isoform_annotated.gtf"
REFERENCE_FASTA="/data/gpfs/projects/punim2251/LongBench_data/reference/GRCh38.primary_assembly.genome.fa"

# Set parameters
THREADS="12"  # Set to match cpus-per-task
MIN_READS="25"  # Increased from default for more reliable estimation
MAX_DEPTH="6"   # Optimized for XGBoost
N_ESTIMATORS="200"  # Increased for better model performance
LEARNING_RATE="0.05"  # Lower learning rate for more careful boosting
SUBSAMPLE="0.8"  # Use 80% of data for each tree
COLSAMPLE="0.8"  # Use 80% of features for each tree

# Load required modules - MUST BE DONE BEFORE ANY COMMAND THAT USES THEM
source /data/gpfs/projects/punim2251/sirv-integration-pipeline/sirv_env/bin/activate
pip install -e .
echo "Loading required modules..."
module load GCCcore/11.3.0
module load GCC/11.3.0
module load SAMtools/1.21

# Check if input files exist
if [ ! -f "${FLAMES_BAM}" ]; then
    echo "ERROR: FLAMES BAM file not found at ${FLAMES_BAM}"
    exit 1
fi

if [ ! -f "${FLAMES_GTF}" ]; then
    echo "ERROR: FLAMES GTF file not found at ${FLAMES_GTF}"
    exit 1
fi

if [ ! -f "${REFERENCE_FASTA}" ]; then
    echo "ERROR: Reference FASTA file not found at ${REFERENCE_FASTA}"
    exit 1
fi

# Process FLAMES BAM
if [ -f "${BASE_OUTPUT_DIR}/fixed_flames.bam" ]; then
    FIXED_FLAMES_BAM="${BASE_OUTPUT_DIR}/fixed_flames.bam"
    echo "Using existing fixed FLAMES BAM: ${FIXED_FLAMES_BAM}"
else
    # Create a fixed copy of the FLAMES BAM with proper index
    FIXED_FLAMES_BAM="${BASE_OUTPUT_DIR}/fixed_flames.bam"
    echo "Creating fixed FLAMES BAM at ${FIXED_FLAMES_BAM}..."
    
    # Check that samtools is available
    if ! command -v samtools &> /dev/null; then
        echo "ERROR: samtools command not found. Module loading may have failed."
        exit 1
    fi
    
    samtools view -h ${FLAMES_BAM} | samtools sort -o ${FIXED_FLAMES_BAM}
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create fixed FLAMES BAM file"
        exit 1
    fi
    
    samtools index ${FIXED_FLAMES_BAM}
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to index fixed FLAMES BAM file"
        exit 1
    fi
    
    # Verify the file was created
    if [ ! -f "${FIXED_FLAMES_BAM}" ]; then
        echo "ERROR: Failed to create fixed FLAMES BAM file"
        exit 1
    fi
    echo "Successfully created fixed FLAMES BAM file"
fi

# Verify BAM file exists before proceeding
if [ ! -f "${FIXED_FLAMES_BAM}" ]; then
    echo "ERROR: Fixed FLAMES BAM file not found at ${FIXED_FLAMES_BAM}"
    exit 1
fi

# Change to pipeline directory and activate virtual environment
cd ${PIPELINE_DIR}
source sirv_env/bin/activate

# Install/update the package and required dependencies
echo "Installing/updating the SIRV integration pipeline package and ML dependencies..."
pip install -e .
pip install scikit-learn>=1.0.0 pyfaidx biopython matplotlib>=3.5.0 pandas>=1.0.0 pysam seaborn xgboost>=1.7.0

# Run the evaluation script directly with XGBoost parameters
echo "Starting XGBoost coverage model evaluation with optimized settings..."
python ${PIPELINE_DIR}/evaluate_rf_coverage.py \
    --bam-file ${FIXED_FLAMES_BAM} \
    --gtf-file ${FLAMES_GTF} \
    --output-dir ${BASE_OUTPUT_DIR} \
    --model-type xgboost \
    --min-reads ${MIN_READS} \
    --length-bins 5 \
    --n-estimators ${N_ESTIMATORS} \
    --max-depth ${MAX_DEPTH} \
    --learning-rate ${LEARNING_RATE} \
    --subsample ${SUBSAMPLE} \
    --colsample-bytree ${COLSAMPLE} \
    --feature-extraction \
    --reference-file ${REFERENCE_FASTA} \
    --seed 42 \
    --threads ${THREADS} \
    --verbose

# Copy this script to the output directory for reproducibility
cp $0 ${BASE_OUTPUT_DIR}/

echo "XGBoost coverage model evaluation complete!"
echo "Results are available in: ${BASE_OUTPUT_DIR}"
echo "Comprehensive report: ${BASE_OUTPUT_DIR}/xgb_coverage_report.md"

# Deactivate virtual environment
deactivate 