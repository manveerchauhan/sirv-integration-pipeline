#!/bin/bash
#SBATCH --partition="sapphire"
#SBATCH --nodes=1
#SBATCH --account="punim2251"
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mschauhan@student.unimelb.edu.au
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=1-12:00:00
#SBATCH --job-name="SIRV_Integration"
#SBATCH --output=sirv_integration_%j.out
#SBATCH --error=sirv_integration_%j.err

# Define pipeline directory
PIPELINE_DIR="/data/gpfs/projects/punim2251/sirv-integration-pipeline"

# Define input files and parameters
SIRV_BAM="/data/gpfs/projects/punim0646/ric/PMbrain_cDNAPCR/sirv_bams/bc01_sirv_reads.bam"
SIRV_REF="/data/gpfs/projects/punim0646/sirv_genomes/sirv_transcriptome_c.fa"
SIRV_GTF="/data/gpfs/projects/punim0646/sirv_genomes/sirv_annotation_c.gtf"
SC_FASTQ="/data/gpfs/projects/punim2251/Aim1_LongBench/ReadRarefaction_wFixedCells/data/LongBench_All/ont_sc/10Percent_FLAMES/matched_reads.fastq"
GENOME_REF="/data/gpfs/projects/punim2251/LongBench_data/reference/gencode.v44.transcripts.fa"
OUTPUT_DIR="/data/gpfs/projects/punim2251/test_run"
INSERTION_RATE="0.01"
THREADS="16"  # Match cpus-per-task for optimal thread usage

# Performance tuning
export OMP_NUM_THREADS=${THREADS}
export NUMEXPR_MAX_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}
export OPENBLAS_NUM_THREADS=${THREADS}

# Load required modules
module purge  # Clean environment first
module load GCCcore/11.3.0
module load minimap2/2.26
module load GCC/11.3.0
module load SAMtools/1.21
module load Python/3.10.4  # Ensure consistent Python version

# Change to pipeline directory and activate virtual environment
cd ${PIPELINE_DIR}
source sirv_env/bin/activate

# Performance monitoring
echo "Job started at: $(date)"
START_TIME=$(date +%s)

# Install dependencies efficiently
pip install --no-cache-dir -e .
pip install --no-cache-dir seaborn

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run the pipeline with optimized thread usage
python -m sirv_pipeline --integration \
    --sirv-bam ${SIRV_BAM} \
    --sirv-reference ${SIRV_REF} \
    --sirv-gtf ${SIRV_GTF} \
    --sc-fastq ${SC_FASTQ} \
    --non-sirv-reference ${GENOME_REF} \
    --create-combined-reference \
    --output-dir ${OUTPUT_DIR} \
    --insertion-rate ${INSERTION_RATE} \
    --threads ${THREADS} \
    --verbose

# Performance monitoring
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
echo "Job completed at: $(date)"
echo "Total runtime: $((RUNTIME / 3600)) hours, $(((RUNTIME % 3600) / 60)) minutes, $((RUNTIME % 60)) seconds"

# Deactivate virtual environment
deactivate 