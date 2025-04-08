#!/usr/bin/env python
"""
Example script demonstrating how to use the SIRV Integration Pipeline
with the specific mapping parameters from the SLURM script.

This script runs the SIRV mapping and creates alignment files
using the minimap2 parameters: -ax map-ont -t 8 --sam-hit-only --secondary=no
"""

import os
import sys
import argparse
import logging

# Add parent directory to path to import sirv_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirv_pipeline.mapping import create_alignment, map_sirv_reads
from sirv_pipeline.utils import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Run SIRV mapping with specific parameters")
    
    parser.add_argument("--sirv-fastq", required=True, help="Path to SIRV FASTQ file")
    parser.add_argument("--sirv-reference", required=True, help="Path to SIRV reference FASTA file")
    parser.add_argument("--sirv-gtf", required=True, help="Path to SIRV GTF annotation file")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (default: 8)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "sirv_mapping.log")
    setup_logger(log_file, console_level=logging.INFO, file_level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting SIRV mapping with specific parameters")
    
    # Output files
    transcript_map_file = os.path.join(args.output_dir, "transcript_map.csv")
    alignment_file = os.path.join(args.output_dir, "sirv_alignment.bam")
    
    # Map SIRV reads to identify transcripts
    logger.info(f"Mapping SIRV reads to identify transcripts: {args.sirv_fastq}")
    map_sirv_reads(
        args.sirv_fastq,
        args.sirv_reference,
        args.sirv_gtf,
        transcript_map_file,
        threads=args.threads,
        keep_temp=True  # Keep temporary files for inspection
    )
    
    # Create alignment file with specific parameters
    logger.info(f"Creating SIRV alignment: {args.sirv_fastq}")
    create_alignment(
        args.sirv_fastq,
        args.sirv_reference,
        alignment_file,
        threads=args.threads,
        preset="map-ont"  # Use Oxford Nanopore preset
    )
    
    logger.info(f"SIRV mapping completed. Results in {args.output_dir}")
    logger.info(f"Transcript map: {transcript_map_file}")
    logger.info(f"Alignment file: {alignment_file}")

if __name__ == "__main__":
    main() 