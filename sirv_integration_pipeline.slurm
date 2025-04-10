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
#SBATCH --mem=40000
#SBATCH --time=0-03:00:00
#SBATCH --job-name="SIRV_Integration"
#SBATCH --output=sirv_integration_%j.out
#SBATCH --error=sirv_integration_%j.err

# Define pipeline directory
PIPELINE_DIR="/data/gpfs/projects/punim2251/sirv-integration-pipeline"

# Define input files and parameters
SIRV_BAM="/data/gpfs/projects/punim0646/ric/PMbrain_cDNAPCR/sirv_bams/bc01_sirv_reads.bam"
SIRV_REF="/data/gpfs/projects/punim0646/sirv_genomes/sirv_transcriptome_c.fa"
SC_FASTQ="/data/gpfs/projects/punim2251/Aim1_LongBench/ReadRarefaction_wFixedCells/data/LongBench_All/ont_sc/10Percent_FLAMES/matched_reads.fastq"
FLAMES_BAM="/data/gpfs/projects/punim2251/Aim1_LongBench/ReadRarefaction_wFixedCells/data/LongBench_All/ont_sc/10Percent_FLAMES/realign2transcript.bam"
OUTPUT_DIR="/data/gpfs/projects/punim2251/sirv_run_01_longbench"
INSERTION_RATE="0.01"
THREADS="12"  # Set to match cpus-per-task

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Load required modules
module load GCCcore/11.3.0
module load minimap2/2.26
module load GCC/11.3.0
module load SAMtools/1.21

# Check if FLAMES BAM file exists
if [ ! -f "${FLAMES_BAM}" ]; then
    echo "ERROR: FLAMES BAM file not found at ${FLAMES_BAM}"
    echo "Please check the path and try again."
    exit 1
fi

# Change to pipeline directory and activate virtual environment
cd ${PIPELINE_DIR}
source sirv_env/bin/activate
echo "Reinstalling the updated pipeline..."
pip install -e .
pip install seaborn

# Resume the pipeline from the coverage modeling step
# We'll skip the previous steps since they completed successfully
echo "Resuming SIRV integration pipeline with custom coverage model from FLAMES output..."
python -m sirv_pipeline --integration \
    --sirv-bam ${SIRV_BAM} \
    --sirv-reference ${SIRV_REF} \
    --sc-fastq ${SC_FASTQ} \
    --output-dir ${OUTPUT_DIR} \
    --insertion-rate ${INSERTION_RATE} \
    --threads ${THREADS} \
    --coverage-model custom \
    --learn-coverage-from ${FLAMES_BAM} \
    --min-reads 5 \
    --length-bins 5 \
    --visualize-coverage \
    --verbose

# Deactivate virtual environment
deactivate

echo "Pipeline run completed." 