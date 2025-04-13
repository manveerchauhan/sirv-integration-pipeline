#!/usr/bin/env python3
"""
Comparative BamSlam Analysis for SIRV Integration

This script generates a comparative report of BamSlam metrics for:
1. Original SIRV data
2. Processed SIRV data (after coverage model)
3. Original scRNA-seq data

The report helps evaluate how well the coverage model transformed
SIRV reads to match the scRNA-seq coverage patterns.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sirv_pipeline.plotting.bamslam_plots import run_bamslam_analysis, import_bam_file
import pysam
from datetime import datetime
import traceback

# Import new modules
from sirv_pipeline.utils.bamslam_utils import (
    setup_bamslam_logger, 
    extract_sirv_reads_from_integrated, 
    map_fastq_to_reference
)
from sirv_pipeline.utils.bamslam_plots import (
    plot_coverage_patterns,
    plot_length_distribution,
    plot_coverage_bias,
    plot_length_stratified_coverage,
    generate_comparison_heatmap
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comparative BamSlam Analysis for SIRV Integration"
    )
    
    parser.add_argument(
        "--sirv-bam", type=str, required=True,
        help="Path to original SIRV BAM file"
    )
    
    parser.add_argument(
        "--integrated-fastq", type=str, required=True,
        help="Path to integrated SIRV reads (FASTQ after processing by coverage model)"
    )
    
    parser.add_argument(
        "--sirv-extracted-fastq", type=str,
        help="Path to extracted SIRV reads (before processing, optional)"
    )
    
    parser.add_argument(
        "--tracking-csv", type=str,
        help="Path to integration tracking CSV file (contains read IDs of processed SIRV reads)"
    )
    
    parser.add_argument(
        "--sc-bam", type=str, required=True,
        help="Path to original scRNA-seq BAM file"
    )
    
    parser.add_argument(
        "--sirv-reference", type=str, required=True,
        help="Path to SIRV reference FASTA file"
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="./comparative_bamslam",
        help="Path to output directory (default: ./comparative_bamslam)"
    )
    
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def analyze_all_datasets(args):
    """
    Analyze all datasets for comparative analysis.
    
    Args:
        args: Command-line arguments
        
    Returns:
        dict: Analysis results for all datasets
    """
    logger = logging.getLogger()
    
    # Create output directories
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Step 1: Extract SIRV reads from integrated FASTQ
    logger.info("Step 1: Extracting SIRV reads from integrated FASTQ")
    processed_sirv_fastq = extract_sirv_reads_from_integrated(
        args.integrated_fastq, 
        temp_dir,
        args.tracking_csv,
        args.sirv_extracted_fastq
    )
    
    # Step 2: Map processed SIRV reads to reference
    logger.info("Step 2: Mapping processed SIRV reads to reference")
    processed_sirv_bam = map_fastq_to_reference(
        processed_sirv_fastq,
        args.sirv_reference,
        temp_dir
    )
    
    if not processed_sirv_bam:
        logger.error("Failed to map processed SIRV reads, cannot continue")
        sys.exit(1)
    
    # Step 3: Run BamSlam analysis on all datasets
    logger.info("Step 3: Running BamSlam analysis on all datasets")
    
    # Output directories for each dataset
    original_sirv_dir = os.path.join(output_dir, "original_sirv")
    processed_sirv_dir = os.path.join(output_dir, "processed_sirv")
    sc_dir = os.path.join(output_dir, "sc")
    
    os.makedirs(original_sirv_dir, exist_ok=True)
    os.makedirs(processed_sirv_dir, exist_ok=True)
    os.makedirs(sc_dir, exist_ok=True)
    
    # Analyze original SIRV
    logger.info("Analyzing original SIRV BAM")
    original_sirv_results = run_bamslam_analysis(
        args.sirv_bam,
        original_sirv_dir,
        sample_size=10000,
        bin_count=100
    )
    
    # Analyze processed SIRV
    logger.info("Analyzing processed SIRV BAM")
    processed_sirv_results = run_bamslam_analysis(
        processed_sirv_bam,
        processed_sirv_dir,
        sample_size=10000,
        bin_count=100
    )
    
    # Analyze scRNA-seq
    logger.info("Analyzing scRNA-seq BAM")
    sc_results = run_bamslam_analysis(
        args.sc_bam,
        sc_dir,
        sample_size=10000,
        bin_count=100
    )
    
    # Return combined results
    return {
        'original_sirv': original_sirv_results,
        'processed_sirv': processed_sirv_results,
        'sc': sc_results,
        'metadata': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_sirv_bam': args.sirv_bam,
            'integrated_fastq': args.integrated_fastq,
            'processed_sirv_bam': processed_sirv_bam,
            'sc_bam': args.sc_bam,
            'sirv_reference': args.sirv_reference
        }
    }

def load_analysis_data(results, output_dir):
    """
    Load additional analysis data from results.
    
    Args:
        results: Analysis results
        output_dir: Output directory
        
    Returns:
        dict: Enhanced analysis data
    """
    logger = logging.getLogger()
    logger.info("Loading additional analysis data")
    
    # Add similarity matrix
    try:
        # Create similarity matrix for coverage patterns
        datasets = ['original_sirv', 'processed_sirv', 'sc']
        dataset_names = ['Original SIRV', 'Processed SIRV', 'scRNA-seq']
        
        similarity = pd.DataFrame(index=dataset_names, columns=dataset_names)
        
        # Calculate cosine similarity between coverage patterns
        for i, dataset1 in enumerate(datasets):
            for j, dataset2 in enumerate(datasets):
                if dataset1 in results and dataset2 in results:
                    coverage1 = results[dataset1].get('coverage_data', {}).get('coverage', [])
                    coverage2 = results[dataset2].get('coverage_data', {}).get('coverage', [])
                    
                    if len(coverage1) > 0 and len(coverage2) > 0:
                        # Calculate cosine similarity
                        similarity.iloc[i, j] = np.dot(coverage1, coverage2) / (
                            np.linalg.norm(coverage1) * np.linalg.norm(coverage2))
                    else:
                        similarity.iloc[i, j] = 0.0
                else:
                    similarity.iloc[i, j] = 0.0
        
        results['similarity_matrix'] = similarity
        
    except Exception as e:
        logger.error(f"Error calculating similarity matrix: {str(e)}")
        logger.error(traceback.format_exc())
    
    return results

def generate_comparative_plots(data, output_dir):
    """
    Generate comparative plots for all datasets.
    
    Args:
        data: Analysis data
        output_dir: Output directory
    """
    logger = logging.getLogger()
    logger.info("Generating comparative plots")
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Coverage pattern comparison
    plot_coverage_patterns(
        data, 
        os.path.join(plots_dir, "coverage_pattern_comparison.png"),
        "Coverage Pattern Comparison"
    )
    
    # 2. Read length distribution
    plot_length_distribution(
        data,
        os.path.join(plots_dir, "length_distribution.png"),
        "Read Length Distribution"
    )
    
    # 3. Coverage bias metrics
    plot_coverage_bias(
        data,
        os.path.join(plots_dir, "coverage_bias_comparison.png"),
        "Coverage Bias Metrics Comparison"
    )
    
    # 4. Length-stratified coverage
    plot_length_stratified_coverage(
        data,
        os.path.join(plots_dir, "length_stratified_coverage.png"),
        "Length-Stratified Coverage Comparison"
    )
    
    # 5. Similarity heatmap
    generate_comparison_heatmap(
        data,
        os.path.join(plots_dir, "similarity_heatmap.png"),
        "Coverage Pattern Similarity"
    )

def generate_comparative_report(data, output_dir, metadata):
    """
    Generate a comprehensive HTML report.
    
    Args:
        data: Analysis data
        output_dir: Output directory
        metadata: Metadata for the report
    """
    logger = logging.getLogger()
    logger.info("Generating comparative report")
    
    # Create report file
    report_file = os.path.join(output_dir, "comparative_report.html")
    
    try:
        with open(report_file, 'w') as f:
            # Write HTML header
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>SIRV Integration Comparative Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        .metric-table th, .metric-table td {{
            border: 1px solid #dee2e6;
            padding: 8px 12px;
            text-align: left;
        }}
        .metric-table th {{
            background-color: #f8f9fa;
        }}
        .metric-row:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot-img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #dee2e6;
            border-radius: 5px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #dee2e6;
            font-size: 0.9em;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SIRV Integration Comparative Analysis Report</h1>
        <div class="summary-box">
            <h3>Analysis Summary</h3>
            <p><strong>Generated:</strong> {metadata.get('timestamp', 'Unknown')}</p>
            <p><strong>Original SIRV BAM:</strong> {metadata.get('original_sirv_bam', 'Unknown')}</p>
            <p><strong>Processed SIRV BAM:</strong> {metadata.get('processed_sirv_bam', 'Unknown')}</p>
            <p><strong>scRNA-seq BAM:</strong> {metadata.get('sc_bam', 'Unknown')}</p>
            <p><strong>SIRV Reference:</strong> {metadata.get('sirv_reference', 'Unknown')}</p>
        </div>
        
        <h2>Coverage Pattern Comparison</h2>
        <p>
            This analysis compares the coverage patterns of original SIRV reads, processed SIRV reads 
            (after applying the coverage model), and scRNA-seq reads. Ideally, the processed SIRV 
            coverage pattern should closely match the scRNA-seq pattern.
        </p>
        <div class="plot-container">
            <img class="plot-img" src="plots/coverage_pattern_comparison.png" alt="Coverage Pattern Comparison">
        </div>
        
        <h2>Coverage Bias Metrics</h2>
        <p>
            The following table shows key coverage bias metrics for each dataset:
        </p>
        <table class="metric-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Original SIRV</th>
                    <th>Processed SIRV</th>
                    <th>scRNA-seq</th>
                </tr>
            </thead>
            <tbody>
""")
            
            # Add metrics to table
            metrics = [
                ('5\' Bias', 'five_prime_bias'),
                ('3\' Bias', 'three_prime_bias'),
                ('Coverage Evenness', 'evenness'),
                ('Coefficient of Variation', 'cv')
            ]
            
            for display_name, metric_name in metrics:
                orig_value = data.get('original_sirv', {}).get('metrics', {}).get(metric_name, 'N/A')
                proc_value = data.get('processed_sirv', {}).get('metrics', {}).get(metric_name, 'N/A')
                sc_value = data.get('sc', {}).get('metrics', {}).get(metric_name, 'N/A')
                
                if isinstance(orig_value, float):
                    orig_value = f"{orig_value:.3f}"
                if isinstance(proc_value, float):
                    proc_value = f"{proc_value:.3f}"
                if isinstance(sc_value, float):
                    sc_value = f"{sc_value:.3f}"
                
                f.write(f"""
                <tr class="metric-row">
                    <td>{display_name}</td>
                    <td>{orig_value}</td>
                    <td>{proc_value}</td>
                    <td>{sc_value}</td>
                </tr>""")
            
            # Write the rest of the report
            f.write(f"""
            </tbody>
        </table>
        
        <div class="plot-container">
            <img class="plot-img" src="plots/coverage_bias_comparison.png" alt="Coverage Bias Comparison">
        </div>
        
        <h2>Coverage Pattern Similarity</h2>
        <p>
            The heatmap below shows the similarity between coverage patterns. A value of 1.0 indicates 
            identical coverage patterns, while values close to 0 indicate very different patterns.
        </p>
        <div class="plot-container">
            <img class="plot-img" src="plots/similarity_heatmap.png" alt="Similarity Heatmap">
        </div>
        
        <h2>Read Length Distribution</h2>
        <p>
            The distribution of read lengths for each dataset:
        </p>
        <div class="plot-container">
            <img class="plot-img" src="plots/length_distribution.png" alt="Read Length Distribution">
        </div>
        
        <h2>Length-Stratified Coverage Patterns</h2>
        <p>
            Coverage patterns stratified by read length show how the coverage bias varies across 
            different read length categories:
        </p>
        <div class="plot-container">
            <img class="plot-img" src="plots/length_stratified_coverage.png" alt="Length-Stratified Coverage">
        </div>
        
        <div class="footer">
            <p>Generated by SIRV Integration Pipeline Comparative Analysis Tool</p>
        </div>
    </div>
</body>
</html>
""")
        
        logger.info(f"Comparative report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Error generating comparative report: {str(e)}")
        logger.error(traceback.format_exc())

def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logger
    log_file = os.path.join(args.output_dir, "comparative_analysis.log")
    logger = setup_bamslam_logger(
        log_file=log_file,
        console_level=logging.INFO if not args.verbose else logging.DEBUG,
        file_level=logging.DEBUG
    )
    
    logger.info("Starting Comparative BamSlam Analysis")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Analyze all datasets
        results = analyze_all_datasets(args)
        
        # Load additional analysis data
        data = load_analysis_data(results, args.output_dir)
        
        # Generate comparative plots
        generate_comparative_plots(data, args.output_dir)
        
        # Generate comparative report
        generate_comparative_report(data, args.output_dir, results['metadata'])
        
        logger.info("Comparative analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during comparative analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 