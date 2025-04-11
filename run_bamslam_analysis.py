#!/usr/bin/env python3
"""
SIRV BamSlam Analysis Script

This script runs BamSlam-style analysis on SIRV alignment data.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from sirv_pipeline.plotting.bamslam_plots import run_bamslam_analysis

def setup_logger(log_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """Set up logger for the script."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                           datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SIRV BamSlam Analysis Script"
    )
    
    parser.add_argument(
        "--bam", type=str, required=True,
        help="Path to SIRV BAM file for analysis"
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="./bamslam_output",
        help="Path to output directory (default: ./bamslam_output)"
    )
    
    parser.add_argument(
        "--data-type", type=str, choices=["cdna", "rna"], default="cdna",
        help="Data type, either 'cdna' or 'rna' (default: cdna)"
    )
    
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    log_file = os.path.join(args.output_dir, "bamslam_analysis.log")
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(log_file, console_level=log_level, file_level=logging.DEBUG)
    
    logger.info("Starting SIRV BamSlam Analysis")
    logger.info(f"BAM file: {args.bam}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Data type: {args.data_type}")
    
    # Run the analysis
    try:
        results = run_bamslam_analysis(
            bam_file=args.bam,
            output_dir=args.output_dir,
            data_type=args.data_type
        )
        
        if results.get("success", False):
            logger.info("BamSlam analysis completed successfully")
            
            # Print some results
            stats = results.get("stats", {})
            if stats:
                logger.info(f"Total reads: {stats.get('total_reads', 'N/A')}")
                logger.info(f"Full-length reads: {stats.get('full_length_reads', 'N/A')} ({stats.get('full_length_percentage', 'N/A'):.2f}%)")
                logger.info(f"Median coverage: {stats.get('median_coverage', 'N/A'):.4f}")
                logger.info(f"Unique transcripts: {stats.get('unique_transcripts', 'N/A')}")
            
            logger.info(f"All outputs saved to: {args.output_dir}")
            logger.info("Check the 'plots' directory for visualizations")
            
            return 0
        else:
            logger.error(f"BamSlam analysis failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Error running BamSlam analysis: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 