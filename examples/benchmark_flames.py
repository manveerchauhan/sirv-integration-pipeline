#!/usr/bin/env python3
"""
Example script for benchmarking FLAMES with SIRV spike-ins.

Author: Manveer Chauhan
Email: mschauhan@student.unimelb.edu.au

This script performs all three steps:
1. Map SIRV reads to reference
2. Add SIRV reads to scRNA-seq dataset
3. Compare with FLAMES output

Example usage:
    python benchmark_flames.py \
        --sirv_fastq data/sirvs.fastq \
        --sc_fastq data/ont_data.fastq \
        --sirv_reference data/sirv_genome.fa \
        --sirv_gtf data/sirv.gtf \
        --flames_output results/flames_output.csv \
        --output_dir ./benchmark_results
"""

import os
import argparse
import subprocess
from sirv_pipeline import (
    map_sirv_reads, 
    add_sirv_to_dataset, 
    compare_with_flames,
    generate_report,
    setup_logger
)

def run_flames(fastq_file, reference_genome, reference_gtf, output_dir, threads=4):
    """
    Run FLAMES on the combined dataset.
    
    NOTE: This is a placeholder function. You would need to adapt this to your
    specific FLAMES installation and parameters.
    """
    logger.info("Running FLAMES on combined dataset...")
    
    # Create FLAMES output directory
    flames_dir = os.path.join(output_dir, "flames_results")
    os.makedirs(flames_dir, exist_ok=True)
    
    # Example FLAMES command - replace with actual command
    flames_cmd = [
        "flames",
        "--reads", fastq_file,
        "--genome", reference_genome,
        "--gtf", reference_gtf,
        "--outdir", flames_dir,
        "--threads", str(threads)
    ]
    
    try:
        logger.info(f"Running command: {' '.join(flames_cmd)}")
        subprocess.run(flames_cmd, check=True)
        logger.info("FLAMES completed successfully")
        
        # Return path to FLAMES output CSV
        # Adjust this path to match your FLAMES output
        flames_output = os.path.join(flames_dir, "transcript_count.csv")
        
        if os.path.exists(flames_output):
            return flames_output
        else:
            logger.error(f"FLAMES output not found: {flames_output}")
            return None
    
    except subprocess.CalledProcessError as e:
        logger.error(f"FLAMES execution failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error running FLAMES: {e}")
        return None


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Benchmark FLAMES with SIRV spike-ins')
    parser.add_argument('--sirv_fastq', required=True, help='SIRV FASTQ file')
    parser.add_argument('--sc_fastq', required=True, help='scRNA-seq FASTQ file')
    parser.add_argument('--sirv_reference', required=True, help='SIRV reference genome')
    parser.add_argument('--sirv_gtf', required=True, help='SIRV annotation GTF')
    parser.add_argument('--reference_genome', required=True, help='Reference genome for FLAMES')
    parser.add_argument('--reference_gtf', required=True, help='Reference GTF for FLAMES')
    parser.add_argument('--flames_output', help='Existing FLAMES output (skip FLAMES run if provided)')
    parser.add_argument('--output_dir', default='./benchmark_results', help='Output directory')
    parser.add_argument('--insertion_rate', type=float, default=0.01, help='SIRV insertion rate')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--run_flames', action='store_true', help='Run FLAMES (requires FLAMES to be installed)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up global logger
    global logger
    logger = setup_logger(
        name="benchmark",
        level=20,  # INFO
        log_file=os.path.join(args.output_dir, "benchmark.log")
    )
    
    logger.info("Starting SIRV benchmark pipeline")
    
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
    
    # Step 3: Run FLAMES if requested
    if args.run_flames and not args.flames_output:
        logger.info("Step 3: Running FLAMES on combined dataset")
        flames_output = run_flames(
            fastq_file=output_fastq,
            reference_genome=args.reference_genome,
            reference_gtf=args.reference_gtf,
            output_dir=args.output_dir,
            threads=args.threads
        )
    else:
        flames_output = args.flames_output
    
    # Step 4: Compare with FLAMES output
    if flames_output and os.path.exists(flames_output):
        logger.info("Step 4: Comparing with FLAMES output")
        
        comparison_file = os.path.join(args.output_dir, "comparison.csv")
        plot_dir = os.path.join(args.output_dir, "plots")
        
        compare_with_flames(
            expected_file=expected_file,
            flames_output=flames_output,
            output_file=comparison_file,
            plot_dir=plot_dir
        )
        
        # Generate HTML report
        html_report = os.path.join(args.output_dir, "report.html")
        generate_report(
            comparison_file=comparison_file,
            output_html=html_report
        )
        
        logger.info(f"Benchmark results saved to: {args.output_dir}")
        logger.info(f"HTML report: {html_report}")
    else:
        logger.warning("Skipping comparison: FLAMES output not available")
    
    logger.info("Benchmark pipeline completed successfully")

if __name__ == "__main__":
    main()