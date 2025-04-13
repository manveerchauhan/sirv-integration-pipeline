#!/bin/bash

# Resume SIRV pipeline from existing run
echo "Resuming SIRV pipeline from existing run"

# Define paths
EXISTING_RUN="/data/gpfs/projects/punim2251/sirv_run_20250413_190356"
SIRV_REFERENCE="/data/gpfs/projects/punim2251/sirv-integration-pipeline/test_data/sirv_reference.fa"

# Activate the virtual environment
echo "Activating virtual environment"
source /data/gpfs/projects/punim2251/sirv-integration-pipeline/sirv_env/bin/activate

# Run the pipeline
cd /data/gpfs/projects/punim2251/sirv-integration-pipeline

echo "Using fixed flames BAM from: $EXISTING_RUN/fixed_flames.bam"
echo "Using fixed SIRV BAM from: $EXISTING_RUN/fixed_sirv.bam"
echo "Using SIRV reference: $SIRV_REFERENCE"

# Run the pipeline with a single command to prevent line parsing issues
python run_pipeline_fixed.py --integration --output-dir "$EXISTING_RUN" --sirv-bam "$EXISTING_RUN/fixed_sirv.bam" --learn-coverage-from "$EXISTING_RUN/fixed_flames.bam" --sirv-reference "$SIRV_REFERENCE" --verbose

# Deactivate virtual environment
source deactivate 2>/dev/null || deactivate 2>/dev/null || echo "Note: Could not deactivate virtual environment"

echo "Pipeline resumed successfully!" 