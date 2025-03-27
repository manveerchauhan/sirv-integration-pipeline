#!/usr/bin/env python3
"""
Simple example script for SIRV integration with ONT scRNA-seq data.

Author: Manveer Chauhan
Email: mschauhan@student.unimelb.edu.au
"""

import os
import argparse
from sirv_pipeline import (
    map_sirv_reads, 
    add_sirv_to_dataset, 
    setup_logger
)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Simple SIRV integration example')
    parser.add_argument('--sirv_fastq', required=True, help='SIRV FASTQ file')
    parser.add_argument('--sc_fastq', required=True, help='scRNA-seq FASTQ file')
    parser.add_argument('--sirv_reference', required=True, help='SIRV reference genome')
    parser.add_argument('--sirv_gtf', required=True, help='SIRV annotation GTF')
    parser.add_argument('--output_dir', default='./output', help='Output directory')
    parser.add_argument('--insertion_rate', type=float, default=0.01, 
                      help='SIRV insertion rate (0-0.5)')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logger(
        name="sirv_example",
        level=20,  # INFO
        log_file=os.path.join(args.output_dir, "example.log")
    )
    
    logger.info("Starting SIRV integration example")
    
    # Step 1: Map SIRV reads to reference
    transcript_map = os.path.join(args.output_dir, "transcript_map.csv")
    logger.info("Step 1: Mapping SIRV reads to reference")
    
    map_sirv_reads(
        sirv_fastq=args.sirv_fastq,
        sirv_reference=args.sirv_reference,
        sirv_gtf=args.sirv_gtf,
        output_csv=transcript_map,
        threads=args.threads
    )
    
    # Step 2: Add SIRV reads to scRNA-seq dataset
    output_fastq = os.path.join(args.output_dir, "combined_data.fastq")
    logger.info("Step 2: Adding SIRV reads to scRNA-seq dataset")
    
    output_fastq, tracking_file, expected_file = add_sirv_to_dataset(
        sirv_fastq=args.sirv_fastq,
        sc_fastq=args.sc_fastq,
        sirv_map_csv=transcript_map,
        output_fastq=output_fastq,
        insertion_rate=args.insertion_rate
    )
    
    logger.info(f"Output FASTQ: {output_fastq}")
    logger.info(f"Tracking file: {tracking_file}")
    logger.info(f"Expected counts: {expected_file}")
    logger.info("Example completed successfully")

if __name__ == "__main__":
    main()