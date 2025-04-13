#!/bin/bash
# Resume pipeline from a failed run
#
# This script resumes the SIRV integration pipeline from the 
# sirv_run_01_longbench directory where it previously failed.

set -e

# Set up paths
WORK_DIR="/data/gpfs/projects/punim2251"
SIRV_PIPELINE_DIR="${WORK_DIR}/sirv-integration-pipeline"
SIRV_DATA_DIR="${WORK_DIR}/sirv_run_01_longbench"
FLAMES_DIR="/data/gpfs/projects/punim2251/Aim1_LongBench/ReadRarefaction_wFixedCells/data/LongBench_All/ont_sc/10Percent_FLAMES"
FLAMES_BAM_PATH="${FLAMES_DIR}/realign2transcript.bam"
FLAMES_GTF_PATH="${FLAMES_DIR}/isoform_annotated.gtf"

# Define default values
RESTART=false
FORCE=false
SUBMIT_SLURM=false
OUTPUT_DIR=""
PREVIOUS_JOB=""

# Display usage information
function show_usage {
    echo "Usage: $0 [options] <output_directory>"
    echo ""
    echo "Options:"
    echo "  -r            Restart the pipeline from the beginning (clear state)"
    echo "  -f            Force rerun of all steps (even if they were previously completed)"
    echo "  -s            Submit as a SLURM job instead of running locally"
    echo "  -j <job_id>   Use parameters from a previous SLURM job (specify job ID)"
    echo "  -h            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/output_dir               # Resume from the last successful step"
    echo "  $0 -r /path/to/output_dir            # Restart the pipeline from the beginning"
    echo "  $0 -f -s /path/to/output_dir         # Force rerun all steps and submit as SLURM job"
    echo "  $0 -j 12345 /path/to/new_output_dir  # Use settings from job 12345 but with a new output dir"
}

# Process command line options
while getopts "rfsj:h" opt; do
    case $opt in
        r) RESTART=true ;;
        f) FORCE=true ;;
        s) SUBMIT_SLURM=true ;;
        j) PREVIOUS_JOB="$OPTARG" ;;
        h) show_usage; exit 0 ;;
        *) echo "Invalid option: -$OPTARG"; show_usage; exit 1 ;;
    esac
done

# Shift to remove options
shift $((OPTIND-1))

# Check if output directory is provided
if [ -z "$1" ]; then
    echo "ERROR: Please specify an output directory"
    show_usage
    exit 1
fi

OUTPUT_DIR="$1"

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ] && [ "$RESTART" = false ] && [ -z "$PREVIOUS_JOB" ]; then
    echo "ERROR: Output directory does not exist: $OUTPUT_DIR"
    echo "Use -r to create a new directory for a fresh start"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define paths
PIPELINE_DIR="/data/gpfs/projects/punim2251/sirv-integration-pipeline"
SLURM_SCRIPT="$PIPELINE_DIR/sirv_integration_pipeline.slurm"
PIPELINE_SCRIPT="$PIPELINE_DIR/run_sirv_pipeline.py"

# If using a previous job, extract parameters
if [ -n "$PREVIOUS_JOB" ]; then
    echo "Extracting parameters from previous job: $PREVIOUS_JOB"
    
    # Check if the job output file exists
    JOB_OUTPUT="sirv_integration_${PREVIOUS_JOB}.out"
    if [ ! -f "$JOB_OUTPUT" ]; then
        # Try to find it in the current directory
        if [ ! -f "$PWD/$JOB_OUTPUT" ]; then
            echo "ERROR: Could not find job output file: $JOB_OUTPUT"
            exit 1
        else
            JOB_OUTPUT="$PWD/$JOB_OUTPUT"
        fi
    fi
    
    echo "Found job output file: $JOB_OUTPUT"
    
    # Extract command parameters from job output
    EXTRACTED_CMD=$(grep -o "Running command: python.*" "$JOB_OUTPUT" | tail -1 | sed 's/Running command: //')
    
    if [ -z "$EXTRACTED_CMD" ]; then
        echo "ERROR: Could not extract command from job output"
        exit 1
    fi
    
    echo "Extracted command: $EXTRACTED_CMD"
    
    # Create a temporary script for SLURM submission
    TMP_SCRIPT=$(mktemp)
    
    # Copy the header from the original SLURM script
    grep "^#SBATCH" "$SLURM_SCRIPT" > "$TMP_SCRIPT"
    
    # Add the command with the new output directory
    echo "" >> "$TMP_SCRIPT"
    echo "# Define pipeline directory" >> "$TMP_SCRIPT"
    echo "PIPELINE_DIR=\"$PIPELINE_DIR\"" >> "$TMP_SCRIPT"
    echo "" >> "$TMP_SCRIPT"
    echo "# Define restart options" >> "$TMP_SCRIPT"
    echo "RESTART=$RESTART" >> "$TMP_SCRIPT"
    echo "FORCE=$FORCE" >> "$TMP_SCRIPT"
    echo "" >> "$TMP_SCRIPT"
    echo "# Change to pipeline directory and activate environment" >> "$TMP_SCRIPT"
    echo "cd \$PIPELINE_DIR" >> "$TMP_SCRIPT"
    echo "source sirv_env/bin/activate" >> "$TMP_SCRIPT"
    echo "" >> "$TMP_SCRIPT"
    echo "# Rerun the pipeline with updated parameters" >> "$TMP_SCRIPT"
    
    # Update the output directory in the command
    MODIFIED_CMD=$(echo "$EXTRACTED_CMD" | sed "s|--output-dir [^ ]*|--output-dir $OUTPUT_DIR|")
    echo "$MODIFIED_CMD" >> "$TMP_SCRIPT"
    
    # Make the script executable
    chmod +x "$TMP_SCRIPT"
    
    if [ "$SUBMIT_SLURM" = true ]; then
        echo "Submitting SLURM job with parameters from job $PREVIOUS_JOB"
        sbatch "$TMP_SCRIPT"
    else
        echo "Running command with parameters from job $PREVIOUS_JOB"
        bash "$TMP_SCRIPT"
    fi
    
    # Clean up
    rm "$TMP_SCRIPT"
    
    exit 0
fi

# Run the pipeline with the specified options
if [ "$SUBMIT_SLURM" = true ]; then
    # Create SLURM submission options
    SLURM_OPTS=""
    if [ "$RESTART" = true ]; then
        SLURM_OPTS="$SLURM_OPTS -r"
    fi
    if [ "$FORCE" = true ]; then
        SLURM_OPTS="$SLURM_OPTS -f"
    fi
    
    echo "Submitting SIRV integration pipeline to SLURM with options: $SLURM_OPTS"
    sbatch $SLURM_OPTS "$SLURM_SCRIPT"
    exit 0
fi

# Check for required files before running
echo "Checking for required files..."

# Check if the FLAMES BAM file exists
if [ -f "${FLAMES_BAM_PATH}" ]; then
    echo "Found FLAMES BAM file: ${FLAMES_BAM_PATH}"
else
    echo "Warning: FLAMES BAM file not found: ${FLAMES_BAM_PATH}"
    echo "Will try to proceed without it"
    FLAMES_BAM_PATH=""
fi

# Check if the FLAMES GTF file exists
if [ -f "${FLAMES_GTF_PATH}" ]; then
    echo "Found FLAMES GTF file: ${FLAMES_GTF_PATH}"
else
    echo "Warning: FLAMES GTF file not found: ${FLAMES_GTF_PATH}"
    echo "Will try to proceed without it (using SIRV GTF instead for modeling)"
    FLAMES_GTF_PATH=""
fi

# Check if sirv_alignment.bam exists
if [ -f "${SIRV_DATA_DIR}/sirv_alignment.bam" ]; then
    echo "Found SIRV alignment file: ${SIRV_DATA_DIR}/sirv_alignment.bam"
else
    echo "Error: SIRV alignment file not found: ${SIRV_DATA_DIR}/sirv_alignment.bam"
    exit 1
fi

# Check if local_reference.fa exists
if [ -f "${SIRV_DATA_DIR}/local_reference.fa" ]; then
    echo "Found SIRV reference file: ${SIRV_DATA_DIR}/local_reference.fa"
else
    echo "Error: SIRV reference file not found: ${SIRV_DATA_DIR}/local_reference.fa"
    exit 1
fi

# Check if auto_generated_reference.gtf exists
if [ -f "${SIRV_DATA_DIR}/auto_generated_reference.gtf" ]; then
    echo "Found SIRV GTF file: ${SIRV_DATA_DIR}/auto_generated_reference.gtf"
else
    echo "Error: SIRV GTF file not found: ${SIRV_DATA_DIR}/auto_generated_reference.gtf"
    exit 1
fi

echo ""
echo "Starting pipeline run..."
echo ""

# Activate the Python environment
if [ -d "${PIPELINE_DIR}/sirv_env" ]; then
    echo "Activating Python environment..."
    source "${PIPELINE_DIR}/sirv_env/bin/activate"
fi

# Run the pipeline script with appropriate parameters
if [ -n "${FLAMES_BAM_PATH}" ]; then
    # Prepare the annotation file parameter
    if [ -n "${FLAMES_GTF_PATH}" ]; then
        # If FLAMES GTF is available, use it for coverage modeling
        ANNOTATION_PARAM="--flames-gtf ${FLAMES_GTF_PATH}"
    else
        ANNOTATION_PARAM=""
    fi

    # Run pipeline with FLAMES BAM
    echo "Running pipeline with FLAMES BAM and coverage modeling..."
    python "${PIPELINE_SCRIPT}" \
        --integration \
        --output-dir "${OUTPUT_DIR}" \
        --threads 8 \
        --sirv-bam "${SIRV_DATA_DIR}/sirv_alignment.bam" \
        --sirv-reference "${SIRV_DATA_DIR}/local_reference.fa" \
        --sirv-gtf "${SIRV_DATA_DIR}/auto_generated_reference.gtf" \
        --learn-coverage-from "${FLAMES_BAM_PATH}" \
        ${ANNOTATION_PARAM} \
        --coverage-model custom \
        --visualize-coverage \
        --min-reads 50 \
        --length-bins 5 \
        --run-comparative-analysis \
        --verbose
else
    # Run pipeline without FLAMES BAM
    echo "Running pipeline without FLAMES BAM (using default coverage model)..."
    python "${PIPELINE_SCRIPT}" \
        --integration \
        --output-dir "${OUTPUT_DIR}" \
        --threads 8 \
        --sirv-bam "${SIRV_DATA_DIR}/sirv_alignment.bam" \
        --sirv-reference "${SIRV_DATA_DIR}/local_reference.fa" \
        --sirv-gtf "${SIRV_DATA_DIR}/auto_generated_reference.gtf" \
        --coverage-model default \
        --visualize-coverage \
        --run-comparative-analysis \
        --verbose
fi

# Check return code
if [ $? -eq 0 ]; then
    echo ""
    echo "Pipeline run completed successfully"
else
    echo ""
    echo "Pipeline run failed with error code $?"
    exit 1
fi

echo ""
echo "Pipeline completed at: $(date)"

# Resume SIRV pipeline from existing run
echo "Resuming SIRV pipeline from existing run"

# Define paths
EXISTING_RUN="/data/gpfs/projects/punim2251/sirv_run_20250413_190356"
SIRV_REFERENCE="/data/gpfs/projects/punim2251/sirv-integration-pipeline/test_data/sirv_reference.fa"

# Run the pipeline
cd /data/gpfs/projects/punim2251/sirv-integration-pipeline

echo "Using fixed flames BAM from: $EXISTING_RUN/fixed_flames.bam"
echo "Using fixed SIRV BAM from: $EXISTING_RUN/fixed_sirv.bam"
echo "Using SIRV reference: $SIRV_REFERENCE"

python run_pipeline_fixed.py \
  --integration \
  --output-dir "$EXISTING_RUN" \
  --sirv-bam "$EXISTING_RUN/fixed_sirv.bam" \
  --learn-coverage-from "$EXISTING_RUN/fixed_flames.bam" \
  --sirv-reference "$SIRV_REFERENCE" \
  --verbose

echo "Pipeline resumed successfully!" 