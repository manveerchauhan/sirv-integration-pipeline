#!/usr/bin/env bash
# This script runs the SIRV Integration Pipeline using real scRNA-seq data
# with synthetic SIRV reads for testing

# Path to your real scRNA-seq data
REAL_SCRNA_PATH="/data/gpfs/projects/punim2251/Aim1_LongBench/ReadRarefaction_wFixedCells/data/LongBench_All/ont_sc/10Percent_FLAMES/matched_reads_dedup.fastq"

# Generate synthetic SIRV data only (skipping synthetic scRNA-seq)
echo "Generating synthetic SIRV test data..."
python synthetic_sirv_fastq.py \
    --sirv_reads 1000 \
    --output_dir ./test_data 
    
# Run the simple integration with real scRNA-seq data
echo "Running SIRV integration with real scRNA-seq data..."
python /data/gpfs/projects/punim2251/sirv-integration-pipeline/examples/simple_integration.py \
    --sirv_fastq ./test_data/synthetic_sirv.fastq \
    --sc_fastq "$REAL_SCRNA_PATH" \
    --sirv_reference ./test_data/synthetic_sirv.fa \
    --sirv_gtf ./test_data/synthetic_sirv.gtf \
    --output_dir ./output_real_data \
    --insertion_rate 0.01 \
    --threads 4

echo "Done! Check the output_real_data directory for results."
echo ""
echo "The pipeline has generated:"
echo "1. Combined FASTQ file with synthetic SIRV reads added to your real scRNA-seq data"
echo "2. Tracking file showing which SIRV reads were added to which cells"
echo "3. Expected counts file for comparing with tools like FLAMES"
echo ""
echo "Next steps:"
echo "1. Run FLAMES on the combined data (matched_reads_dedup+sirv.fastq)"
echo "2. Compare FLAMES output with the expected_counts.csv file"