"""
SIRV Integration Pipeline - main module

This is the entry point for the SIRV Integration Pipeline, which handles
command-line arguments and coordinates the pipeline's execution.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from typing import Dict, List, Optional, Union

from sirv_pipeline.mapping import map_sirv_reads, create_alignment, process_sirv_bams, extract_fastq_from_bam
from sirv_pipeline.coverage_bias import model_transcript_coverage
from sirv_pipeline.integration import add_sirv_to_dataset
from sirv_pipeline.evaluation import compare_with_flames, generate_report
from sirv_pipeline.utils import setup_logger, check_dependencies, validate_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SIRV Integration Pipeline."""
    parser = argparse.ArgumentParser(
        description="SIRV Integration Pipeline for benchmarking long-read RNA-seq analysis methods"
    )
    
    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument(
        "--integration", action="store_true",
        help="Run in integration mode to add SIRV reads to scRNA-seq dataset"
    )
    mode_group.add_argument(
        "--evaluation", action="store_true",
        help="Run in evaluation mode to compare with FLAMES output"
    )
    
    # Integration mode arguments
    integration_group = parser.add_argument_group("Integration Mode")
    integration_group.add_argument(
        "--sirv-fastq", type=str,
        help="Path to SIRV FASTQ file"
    )
    integration_group.add_argument(
        "--sirv-bam", type=str, nargs='+',
        help="Path to SIRV BAM file(s) (can specify multiple files)"
    )
    integration_group.add_argument(
        "--sirv-reference", type=str,
        help="Path to SIRV reference FASTA file"
    )
    integration_group.add_argument(
        "--sirv-gtf", type=str,
        help="Path to SIRV GTF annotation file"
    )
    integration_group.add_argument(
        "--sc-fastq", type=str,
        help="Path to single-cell FASTQ file"
    )
    integration_group.add_argument(
        "--insertion-rate", type=float, default=0.1,
        help="SIRV insertion rate (0-1, default: 0.1)"
    )
    
    # Evaluation mode arguments
    evaluation_group = parser.add_argument_group("Evaluation Mode")
    evaluation_group.add_argument(
        "--expected-file", type=str,
        help="Path to expected SIRV counts file (from integration mode)"
    )
    evaluation_group.add_argument(
        "--flames-output", type=str,
        help="Path to FLAMES output file"
    )
    
    # Common arguments
    common_group = parser.add_argument_group("Common Settings")
    common_group.add_argument(
        "--output-dir", type=str, default="./output",
        help="Path to output directory (default: ./output)"
    )
    common_group.add_argument(
        "--log-file", type=str,
        help="Path to log file (default: <output_dir>/pipeline.log)"
    )
    common_group.add_argument(
        "--threads", type=int, default=8,
        help="Number of threads for parallel processing (default: 8)"
    )
    common_group.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Check if either integration or evaluation mode is specified
    if not (args.integration or args.evaluation):
        parser.error("At least one mode (--integration or --evaluation) must be specified")
    
    # Check required arguments for integration mode
    if args.integration:
        if not all([args.sirv_reference, args.sirv_gtf, args.sc_fastq]):
            parser.error("Integration mode requires --sirv-reference, --sirv-gtf, and --sc-fastq")
        
        # Ensure either --sirv-fastq or --sirv-bam is provided but not both
        if not (args.sirv_fastq or args.sirv_bam):
            parser.error("Integration mode requires either --sirv-fastq or --sirv-bam")
        if args.sirv_fastq and args.sirv_bam:
            parser.error("Cannot specify both --sirv-fastq and --sirv-bam, choose one input format")
    
    # Check required arguments for evaluation mode
    if args.evaluation and not all([args.expected_file, args.flames_output]):
        parser.error("Evaluation mode requires --expected-file and --flames-output")
    
    # Set default output directory and log file
    if not args.output_dir:
        args.output_dir = "./output"
    
    if not args.log_file:
        args.log_file = os.path.join(args.output_dir, "pipeline.log")
    
    return args


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the SIRV Integration Pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(args.log_file, console_level=log_level, file_level=logging.DEBUG)
    
    # Get logger
    logger = logging.getLogger(__name__)
    logger.info("Starting SIRV Integration Pipeline")
    
    # Check dependencies
    logger.info("Checking dependencies...")
    check_dependencies()
    
    # Determine mode
    integration_mode = args.integration
    evaluation_mode = args.evaluation
    
    # Paths for output files
    transcript_map_file = os.path.join(args.output_dir, "transcript_map.csv")
    coverage_model_file = os.path.join(args.output_dir, "coverage_model.csv")
    integrated_fastq = os.path.join(args.output_dir, "integrated.fastq")
    tracking_file = os.path.join(args.output_dir, "tracking.csv")
    comparison_file = os.path.join(args.output_dir, "comparison.csv")
    report_file = os.path.join(args.output_dir, "report.html")
    
    # Run integration mode
    if integration_mode:
        logger.info("Running in integration mode")
        
        # Define alignment file path
        alignment_file = os.path.join(args.output_dir, "sirv_alignment.bam")
        
        # Use either FASTQ or BAM input
        if args.sirv_fastq:
            # Original FASTQ workflow
            logger.info("Mapping SIRV reads from FASTQ...")
            map_sirv_reads(
                args.sirv_fastq,
                args.sirv_reference,
                args.sirv_gtf,
                transcript_map_file,
                threads=args.threads
            )
            
            # Create alignment
            logger.info("Creating SIRV alignment...")
            create_alignment(
                args.sirv_fastq,
                args.sirv_reference,
                alignment_file,
                threads=args.threads,
                preset="map-ont"
            )
        else:
            # BAM input workflow
            logger.info(f"Processing SIRV reads from {len(args.sirv_bam)} BAM file(s)...")
            process_sirv_bams(
                args.sirv_bam,
                args.sirv_reference,
                args.sirv_gtf,
                transcript_map_file,
                alignment_file,
                threads=args.threads
            )
        
        # Model transcript coverage
        logger.info("Modeling transcript coverage...")
        model_transcript_coverage(
            alignment_file,
            args.sirv_gtf,
            coverage_model_file
        )
        
        # Check if we have any mapped SIRV reads before proceeding
        if os.path.exists(transcript_map_file) and os.path.getsize(transcript_map_file) > 0:
            sirv_map_df = pd.read_csv(transcript_map_file)
            if sirv_map_df.empty:
                logger.warning("No SIRV reads were mapped to transcripts. Skipping integration.")
                logger.info(f"Integration completed with warnings. Output files in {args.output_dir}")
                return
        
        # Add SIRV reads to scRNA-seq dataset
        logger.info("Adding SIRV reads to scRNA-seq dataset...")
        # Create temporary FASTQ from BAM if needed
        sirv_fastq_for_integration = args.sirv_fastq
        if not sirv_fastq_for_integration:
            sirv_fastq_for_integration = os.path.join(args.output_dir, "sirv_extracted.fastq")
            logger.info(f"Extracting FASTQ from BAM for integration: {sirv_fastq_for_integration}")
            extract_fastq_from_bam(alignment_file, sirv_fastq_for_integration)
        
        add_sirv_to_dataset(
            args.sc_fastq,
            sirv_fastq_for_integration,
            transcript_map_file,
            coverage_model_file,
            integrated_fastq,
            tracking_file,
            insertion_rate=args.insertion_rate
        )
        
        logger.info(f"Integration completed. Output files in {args.output_dir}")
    
    # Run evaluation mode
    if evaluation_mode:
        logger.info("Running in evaluation mode")
        
        # If this is a continuation of integration mode, use the generated tracking file
        expected_file = args.expected_file or tracking_file
        
        # Check if the expected file exists
        if not os.path.exists(expected_file):
            logger.error(f"Expected file not found: {expected_file}")
            sys.exit(1)
        
        # Compare with FLAMES output
        logger.info("Comparing with FLAMES output...")
        compare_with_flames(
            expected_file,
            args.flames_output,
            comparison_file,
            plot_dir=os.path.join(args.output_dir, "plots")
        )
        
        # Generate report
        logger.info("Generating report...")
        generate_report(
            comparison_file,
            report_file
        )
        
        logger.info(f"Evaluation completed. Results in {args.output_dir}")
    
    logger.info("SIRV Integration Pipeline completed successfully")


def main():
    """Entry point for the SIRV Integration Pipeline."""
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()