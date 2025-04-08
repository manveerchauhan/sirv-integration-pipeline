#!/usr/bin/env python
"""
Example script demonstrating how to use the SIRV Integration Pipeline
with BAM files as input instead of FASTQ files.

This script shows how to use the new --sirv-bam parameter to process 
one or more SIRV BAM files for integration with scRNA-seq data.
"""

import os
import sys
import argparse
import logging

# Add parent directory to path to import sirv_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirv_pipeline.mapping import process_sirv_bams, extract_fastq_from_bam
from sirv_pipeline.utils import setup_logger
from sirv_pipeline.integration import add_sirv_to_dataset
from sirv_pipeline.coverage_bias import model_transcript_coverage

def parse_args():
    parser = argparse.ArgumentParser(description="Run SIRV integration with BAM files")
    
    parser.add_argument("--sirv-bam", required=True, nargs='+', help="Path to one or more SIRV BAM files")
    parser.add_argument("--sirv-reference", required=True, help="Path to SIRV reference FASTA file")
    parser.add_argument("--sirv-gtf", required=True, help="Path to SIRV GTF annotation file")
    parser.add_argument("--sc-fastq", required=True, help="Path to single-cell FASTQ file")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--insertion-rate", type=float, default=0.1, help="SIRV insertion rate (default: 0.1)")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (default: 8)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "sirv_bam_integration.log")
    setup_logger(log_file, console_level=logging.INFO, file_level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting SIRV integration with BAM files")
    
    # Output files
    transcript_map_file = os.path.join(args.output_dir, "transcript_map.csv")
    alignment_file = os.path.join(args.output_dir, "sirv_alignment.bam")
    coverage_model_file = os.path.join(args.output_dir, "coverage_model.csv")
    integrated_fastq = os.path.join(args.output_dir, "integrated.fastq")
    tracking_file = os.path.join(args.output_dir, "tracking.csv")
    
    # Process SIRV BAM files
    logger.info(f"Processing {len(args.sirv_bam)} SIRV BAM file(s)")
    process_sirv_bams(
        args.sirv_bam,
        args.sirv_reference,
        args.sirv_gtf,
        transcript_map_file,
        alignment_file,
        threads=args.threads
    )
    
    # Model transcript coverage
    logger.info("Modeling transcript coverage")
    model_transcript_coverage(
        alignment_file,
        args.sirv_gtf,
        coverage_model_file
    )
    
    # Extract FASTQ from BAM for integration
    logger.info("Extracting FASTQ from BAM for integration")
    sirv_fastq = os.path.join(args.output_dir, "sirv_extracted.fastq")
    extract_fastq_from_bam(alignment_file, sirv_fastq)
    
    # Add SIRV reads to scRNA-seq dataset
    logger.info("Adding SIRV reads to scRNA-seq dataset")
    output_fastq, tracking, expected = add_sirv_to_dataset(
        args.sc_fastq,
        sirv_fastq,
        transcript_map_file,
        coverage_model_file,
        integrated_fastq,
        tracking_file,
        insertion_rate=args.insertion_rate
    )
    
    logger.info(f"Integration completed. Results in {args.output_dir}")
    logger.info(f"Integrated FASTQ: {output_fastq}")
    logger.info(f"Tracking file: {tracking}")
    logger.info(f"Expected counts: {expected}")

if __name__ == "__main__":
    main() 