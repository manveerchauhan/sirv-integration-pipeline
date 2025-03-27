"""
Command-line interface for the SIRV Integration Pipeline.

This module provides a command-line interface for mapping SIRV reads 
to reference, integrating them with scRNA-seq data, and evaluating
the results against FLAMES output.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any

from sirv_pipeline.mapping import map_sirv_reads, get_transcript_statistics
from sirv_pipeline.integration import add_sirv_to_dataset
from sirv_pipeline.evaluation import compare_with_flames, generate_report
from sirv_pipeline.utils import setup_logger, check_dependencies, validate_insertion_rate


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='SIRV Integration Pipeline for Long-Read scRNA-seq',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--sirv_fastq', required=True, 
                       help='SIRV bulk FASTQ file')
    required.add_argument('--sc_fastq', required=True, 
                       help='scRNA-seq FASTQ file')
    required.add_argument('--sirv_reference', required=True, 
                       help='SIRV reference genome')
    required.add_argument('--sirv_gtf', required=True, 
                       help='SIRV annotation GTF')
    required.add_argument('--output_fastq', required=True, 
                       help='Output combined FASTQ')
    
    # Optional arguments
    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('--reference_transcriptome', 
                       help='Reference transcriptome (for coverage modeling)')
    optional.add_argument('--flames_output', 
                       help='FLAMES output for comparison')
    optional.add_argument('--insertion_rate', type=float, default=0.01, 
                       help='SIRV insertion rate (0-0.5)')
    optional.add_argument('--threads', type=int, default=4, 
                       help='Number of threads')
    optional.add_argument('--output_dir', 
                       help='Output directory (defaults to output_fastq directory)')
    optional.add_argument('--log_file', 
                       help='Log file path (defaults to output_dir/pipeline.log)')
    optional.add_argument('--min_overlap', type=float, default=0.5,
                       help='Minimum overlap for transcript assignment')
    optional.add_argument('--sample_size', type=int, default=1000,
                       help='Number of reads to sample for modeling')
    optional.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    optional.add_argument('--keep_temp', action='store_true',
                       help='Keep temporary files')
    optional.add_argument('--no_coverage_bias', action='store_true',
                       help='Disable 5\'-3\' coverage bias modeling')
    optional.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    optional.add_argument('--quiet', action='store_true',
                       help='Disable console output')
    
    args = parser.parse_args()
    
    # Derive output_dir from output_fastq if not provided
    if not args.output_dir:
        args.output_dir = os.path.dirname(os.path.abspath(args.output_fastq))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Derive log_file from output_dir if not provided
    if not args.log_file:
        args.log_file = os.path.join(args.output_dir, "pipeline.log")
    
    return args


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Run the SIRV integration pipeline.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(
        name="sirv_pipeline",
        level=log_level,
        log_file=args.log_file,
        console=not args.quiet
    )
    
    logger.info("Starting SIRV Integration Pipeline")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Aborting.")
        sys.exit(1)
    
    try:
        # Validate insertion rate
        args.insertion_rate = validate_insertion_rate(args.insertion_rate)
        
        # Step 1: Map SIRV reads to reference
        transcript_map = os.path.join(args.output_dir, "transcript_map.csv")
        logger.info(f"Step 1: Mapping SIRV reads to reference")
        map_sirv_reads(
            sirv_fastq=args.sirv_fastq,
            sirv_reference=args.sirv_reference,
            sirv_gtf=args.sirv_gtf,
            output_csv=transcript_map,
            threads=args.threads,
            min_overlap=args.min_overlap,
            keep_temp=args.keep_temp
        )
        
        # Print transcript statistics
        stats = get_transcript_statistics(transcript_map)
        logger.info(f"Found {stats['total_reads']} SIRV reads mapping to {stats['unique_transcripts']} unique transcripts")
        
        # Step 2: Add SIRV reads to scRNA-seq dataset
        logger.info(f"Step 2: Adding SIRV reads to scRNA-seq dataset")
        tracking_file = os.path.join(args.output_dir, "tracking.csv")
        expected_file = os.path.join(args.output_dir, "expected_counts.csv")
        
        output_fastq, tracking_file, expected_file = add_sirv_to_dataset(
            sirv_fastq=args.sirv_fastq,
            sc_fastq=args.sc_fastq,
            sirv_map_csv=transcript_map,
            output_fastq=args.output_fastq,
            insertion_rate=args.insertion_rate,
            tracking_file=tracking_file,
            expected_file=expected_file,
            sample_size=args.sample_size,
            reference_file=args.reference_transcriptome,
            model_coverage_bias=not args.no_coverage_bias,
            seed=args.seed
        )
        
        # Step 3: Compare with FLAMES (if provided)
        if args.flames_output:
            logger.info(f"Step 3: Comparing with FLAMES output")
            comparison_file = os.path.join(args.output_dir, "comparison.csv")
            plot_dir = os.path.join(args.output_dir, "plots")
            
            comparison_df = compare_with_flames(
                expected_file=expected_file,
                flames_output=args.flames_output,
                output_file=comparison_file,
                plot_dir=plot_dir
            )
            
            # Generate HTML report
            html_report = os.path.join(args.output_dir, "report.html")
            generate_report(
                comparison_file=comparison_file,
                output_html=html_report
            )
            
            logger.info(f"HTML report generated: {html_report}")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {str(e)}")
        sys.exit(1)


def main() -> None:
    """
    Main entry point for the pipeline.
    """
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()