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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sirv_pipeline.plotting.bamslam_plots import run_bamslam_analysis, import_bam_file
import pysam
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from datetime import datetime
import re
import gzip
import tempfile

# Set up logging
def setup_logger(log_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """Set up logger for the script."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
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

def extract_sirv_reads_from_integrated(integrated_fastq, output_dir, tracking_csv=None, sirv_extracted_fastq=None):
    """
    Extract only SIRV reads from the integrated FASTQ file.
    
    Args:
        integrated_fastq: Path to integrated FASTQ file
        output_dir: Output directory
        tracking_csv: Path to tracking CSV file (if available)
        sirv_extracted_fastq: Path to extracted SIRV reads (before processing)
        
    Returns:
        Path to filtered FASTQ file
    """
    logger = logging.getLogger()
    logger.info("Extracting SIRV reads from integrated FASTQ file")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Output FASTQ path
    filtered_fastq = os.path.join(output_dir, "processed_sirv_reads.fastq")
    
    # Method 1: Use tracking CSV if available
    if tracking_csv and os.path.exists(tracking_csv):
        logger.info(f"Using tracking CSV to identify SIRV reads: {tracking_csv}")
        try:
            # Load tracking CSV
            tracking_df = pd.read_csv(tracking_csv)
            
            # Check if the CSV has the expected columns
            if 'read_id' in tracking_df.columns:
                sirv_read_ids = set(tracking_df['read_id'])
                logger.info(f"Found {len(sirv_read_ids)} SIRV read IDs in tracking CSV")
                
                # Extract reads from integrated FASTQ
                count = 0
                with open(filtered_fastq, 'w') as out_f:
                    # Check if FASTQ is gzipped
                    if integrated_fastq.endswith('.gz'):
                        in_f = gzip.open(integrated_fastq, 'rt')
                    else:
                        in_f = open(integrated_fastq, 'r')
                    
                    # Process FASTQ
                    while True:
                        # Read 4 lines at a time (FASTQ format)
                        header = in_f.readline().strip()
                        if not header:
                            break
                        
                        seq = in_f.readline().strip()
                        plus = in_f.readline().strip()
                        qual = in_f.readline().strip()
                        
                        # Extract read ID from header
                        read_id = header[1:].split()[0]
                        
                        # Check if this is a SIRV read
                        if read_id in sirv_read_ids:
                            out_f.write(f"{header}\n{seq}\n{plus}\n{qual}\n")
                            count += 1
                            
                            # Log progress
                            if count % 10000 == 0:
                                logger.info(f"Extracted {count} SIRV reads")
                    
                    # Close input file
                    in_f.close()
                
                logger.info(f"Extracted {count} SIRV reads to {filtered_fastq}")
                return filtered_fastq
            else:
                logger.warning("Tracking CSV does not have 'read_id' column, trying alternative method")
        except Exception as e:
            logger.error(f"Error using tracking CSV: {str(e)}")
            logger.info("Trying alternative method")
    
    # Method 2: Extract reads with "SIRV" in their ID or Description
    logger.info("Extracting SIRV reads based on read ID patterns")
    try:
        # Define patterns to identify SIRV reads
        sirv_patterns = [
            re.compile(r'SIRV', re.IGNORECASE),
            re.compile(r'SIRVome', re.IGNORECASE)
        ]
        
        # Extract reads from integrated FASTQ
        count = 0
        with open(filtered_fastq, 'w') as out_f:
            # Check if FASTQ is gzipped
            if integrated_fastq.endswith('.gz'):
                in_f = gzip.open(integrated_fastq, 'rt')
            else:
                in_f = open(integrated_fastq, 'r')
            
            # Process FASTQ
            while True:
                # Read 4 lines at a time (FASTQ format)
                header = in_f.readline().strip()
                if not header:
                    break
                
                seq = in_f.readline().strip()
                plus = in_f.readline().strip()
                qual = in_f.readline().strip()
                
                # Check if this is a SIRV read
                is_sirv = any(pattern.search(header) for pattern in sirv_patterns)
                
                if is_sirv:
                    out_f.write(f"{header}\n{seq}\n{plus}\n{qual}\n")
                    count += 1
                    
                    # Log progress
                    if count % 10000 == 0:
                        logger.info(f"Extracted {count} SIRV reads")
            
            # Close input file
            in_f.close()
        
        logger.info(f"Extracted {count} SIRV reads to {filtered_fastq}")
        
        # If we couldn't extract any reads, this method didn't work
        if count == 0:
            logger.warning("Could not identify SIRV reads by patterns, trying alternative method")
        else:
            return filtered_fastq
    except Exception as e:
        logger.error(f"Error extracting SIRV reads by patterns: {str(e)}")
    
    # Method 3: Compare integrated FASTQ with original SIRV FASTQ
    if sirv_extracted_fastq and os.path.exists(sirv_extracted_fastq):
        logger.info(f"Using original SIRV FASTQ to identify reads: {sirv_extracted_fastq}")
        try:
            # Extract original SIRV read IDs
            sirv_read_ids = set()
            
            # Check if FASTQ is gzipped
            if sirv_extracted_fastq.endswith('.gz'):
                in_f = gzip.open(sirv_extracted_fastq, 'rt')
            else:
                in_f = open(sirv_extracted_fastq, 'r')
            
            # Process FASTQ
            line_count = 0
            for line in in_f:
                line_count += 1
                if line_count % 4 == 1:  # Header line
                    read_id = line.strip()[1:].split()[0]
                    sirv_read_ids.add(read_id)
            
            in_f.close()
            
            logger.info(f"Found {len(sirv_read_ids)} original SIRV read IDs")
            
            # Extract matching reads from integrated FASTQ
            count = 0
            with open(filtered_fastq, 'w') as out_f:
                # Check if FASTQ is gzipped
                if integrated_fastq.endswith('.gz'):
                    in_f = gzip.open(integrated_fastq, 'rt')
                else:
                    in_f = open(integrated_fastq, 'r')
                
                # Process FASTQ
                while True:
                    # Read 4 lines at a time (FASTQ format)
                    header = in_f.readline().strip()
                    if not header:
                        break
                    
                    seq = in_f.readline().strip()
                    plus = in_f.readline().strip()
                    qual = in_f.readline().strip()
                    
                    # Extract read ID from header
                    read_id = header[1:].split()[0]
                    
                    # Check if this is a SIRV read
                    if read_id in sirv_read_ids:
                        out_f.write(f"{header}\n{seq}\n{plus}\n{qual}\n")
                        count += 1
                        
                        # Log progress
                        if count % 10000 == 0:
                            logger.info(f"Extracted {count} SIRV reads")
                
                # Close input file
                in_f.close()
            
            logger.info(f"Extracted {count} SIRV reads to {filtered_fastq}")
            
            if count > 0:
                return filtered_fastq
            else:
                logger.warning("Could not extract SIRV reads by matching original read IDs")
        except Exception as e:
            logger.error(f"Error extracting SIRV reads by matching: {str(e)}")
    
    # Method 4: Map integrated FASTQ to SIRV reference and extract mapped reads
    logger.info("Mapping integrated FASTQ to SIRV reference to identify SIRV reads")
    try:
        import subprocess
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Map to SIRV reference
            sam_path = os.path.join(temp_dir, "mapped.sam")
            
            # Run minimap2
            cmd = [
                "minimap2", "-ax", "map-ont", 
                "--secondary=no",
                args.sirv_reference,
                integrated_fastq
            ]
            
            with open(sam_path, "w") as sam_file:
                subprocess.run(cmd, check=True, stdout=sam_file)
            
            # Extract read IDs of mapped reads
            mapped_reads = set()
            with open(sam_path, "r") as sam_file:
                for line in sam_file:
                    if line.startswith("@"):
                        continue
                    
                    fields = line.strip().split("\t")
                    if len(fields) >= 11 and int(fields[1]) & 4 == 0:  # Not unmapped
                        mapped_reads.add(fields[0])
            
            logger.info(f"Found {len(mapped_reads)} reads mapped to SIRV reference")
            
            # Extract mapped reads from integrated FASTQ
            count = 0
            with open(filtered_fastq, 'w') as out_f:
                # Check if FASTQ is gzipped
                if integrated_fastq.endswith('.gz'):
                    in_f = gzip.open(integrated_fastq, 'rt')
                else:
                    in_f = open(integrated_fastq, 'r')
                
                # Process FASTQ
                while True:
                    # Read 4 lines at a time (FASTQ format)
                    header = in_f.readline().strip()
                    if not header:
                        break
                    
                    seq = in_f.readline().strip()
                    plus = in_f.readline().strip()
                    qual = in_f.readline().strip()
                    
                    # Extract read ID from header
                    read_id = header[1:].split()[0]
                    
                    # Check if this is a mapped read
                    if read_id in mapped_reads:
                        out_f.write(f"{header}\n{seq}\n{plus}\n{qual}\n")
                        count += 1
                        
                        # Log progress
                        if count % 10000 == 0:
                            logger.info(f"Extracted {count} SIRV reads")
                
                # Close input file
                in_f.close()
            
            logger.info(f"Extracted {count} SIRV reads to {filtered_fastq}")
            
            if count > 0:
                return filtered_fastq
    except Exception as e:
        logger.error(f"Error mapping and extracting SIRV reads: {str(e)}")
    
    # If all methods failed, use the integrated FASTQ as is
    logger.warning("Could not extract SIRV reads using any method. Using integrated FASTQ as is.")
    return integrated_fastq

def map_fastq_to_reference(fastq_file, reference_file, output_dir, threads=4):
    """
    Map FASTQ file to reference using minimap2.
    
    Args:
        fastq_file: Path to FASTQ file
        reference_file: Path to reference FASTA file
        output_dir: Output directory for BAM file
        threads: Number of threads to use
        
    Returns:
        Path to output BAM file
    """
    logger = logging.getLogger()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Output BAM path
    bam_path = os.path.join(output_dir, "processed_sirv_alignment.bam")
    
    # Check if minimap2 is available
    import subprocess
    try:
        subprocess.run(["which", "minimap2"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        logger.error("minimap2 not found. Please install minimap2.")
        return None
    
    # Map with minimap2
    logger.info(f"Mapping processed SIRV reads to reference: {fastq_file}")
    
    try:
        # Run minimap2
        cmd = [
            "minimap2", "-ax", "map-ont", 
            "-t", str(threads),
            "--secondary=no",
            reference_file,
            fastq_file
        ]
        
        # Create SAM output
        sam_path = os.path.join(output_dir, "temp.sam")
        with open(sam_path, "w") as sam_file:
            subprocess.run(cmd, check=True, stdout=sam_file)
        
        # Convert SAM to BAM
        cmd = ["samtools", "view", "-bS", sam_path, "-o", bam_path]
        subprocess.run(cmd, check=True)
        
        # Sort BAM
        sorted_bam = os.path.join(output_dir, "processed_sirv_alignment.sorted.bam")
        cmd = ["samtools", "sort", bam_path, "-o", sorted_bam]
        subprocess.run(cmd, check=True)
        
        # Index BAM
        cmd = ["samtools", "index", sorted_bam]
        subprocess.run(cmd, check=True)
        
        # Remove temporary files
        os.remove(sam_path)
        os.remove(bam_path)
        
        logger.info(f"Mapping completed: {sorted_bam}")
        
        return sorted_bam
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running mapping command: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during mapping: {e}")
        return None

def analyze_all_datasets(args):
    """
    Run BamSlam analysis on all three datasets.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with analysis results for each dataset
    """
    logger = logging.getLogger()
    
    results = {}
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    sirv_output_dir = os.path.join(args.output_dir, "sirv_original")
    processed_output_dir = os.path.join(args.output_dir, "sirv_processed")
    sc_output_dir = os.path.join(args.output_dir, "sc_original")
    
    # 1. Analyze original SIRV data
    logger.info("Analyzing original SIRV data")
    results['sirv_original'] = run_bamslam_analysis(
        bam_file=args.sirv_bam,
        output_dir=sirv_output_dir,
        data_type="cdna"
    )
    
    # 2. Extract and analyze processed SIRV reads
    logger.info("Extracting and analyzing processed SIRV data")
    
    # Create a directory for extracted reads
    extracted_dir = os.path.join(args.output_dir, "extracted_reads")
    os.makedirs(extracted_dir, exist_ok=True)
    
    # Extract SIRV reads from integrated FASTQ
    processed_fastq = extract_sirv_reads_from_integrated(
        integrated_fastq=args.integrated_fastq,
        output_dir=extracted_dir,
        tracking_csv=args.tracking_csv if hasattr(args, 'tracking_csv') else None,
        sirv_extracted_fastq=args.sirv_extracted_fastq if hasattr(args, 'sirv_extracted_fastq') else None
    )
    
    # Map processed SIRV reads to reference
    processed_bam = map_fastq_to_reference(
        fastq_file=processed_fastq,
        reference_file=args.sirv_reference,
        output_dir=processed_output_dir
    )
    
    if processed_bam:
        results['sirv_processed'] = run_bamslam_analysis(
            bam_file=processed_bam,
            output_dir=processed_output_dir,
            data_type="cdna"
        )
    else:
        logger.error("Failed to analyze processed SIRV data")
        results['sirv_processed'] = {"success": False}
    
    # 3. Analyze original scRNA-seq data
    logger.info("Analyzing original scRNA-seq data")
    results['sc_original'] = run_bamslam_analysis(
        bam_file=args.sc_bam,
        output_dir=sc_output_dir,
        data_type="cdna"
    )
    
    return results

def load_analysis_data(results, output_dir):
    """
    Load analysis data from BamSlam output files.
    
    Args:
        results: Dictionary with analysis results
        output_dir: Output directory
        
    Returns:
        Dictionary with loaded DataFrames
    """
    logger = logging.getLogger()
    data = {}
    
    # Data directories
    sirv_dir = os.path.join(output_dir, "sirv_original")
    processed_dir = os.path.join(output_dir, "sirv_processed")
    sc_dir = os.path.join(output_dir, "sc_original")
    
    # Load data if available
    try:
        if results['sirv_original']['success']:
            data['sirv_original'] = pd.read_csv(os.path.join(sirv_dir, "bamslam_data.csv"))
            data['sirv_original_stats'] = pd.read_csv(os.path.join(sirv_dir, "bamslam_stats.csv"))
            data['sirv_original_tx_stats'] = pd.read_csv(os.path.join(sirv_dir, "bamslam_transcript_stats.csv"))
    except Exception as e:
        logger.error(f"Error loading original SIRV data: {e}")
        data['sirv_original'] = None
    
    try:
        if results['sirv_processed']['success']:
            data['sirv_processed'] = pd.read_csv(os.path.join(processed_dir, "bamslam_data.csv"))
            data['sirv_processed_stats'] = pd.read_csv(os.path.join(processed_dir, "bamslam_stats.csv"))
            data['sirv_processed_tx_stats'] = pd.read_csv(os.path.join(processed_dir, "bamslam_transcript_stats.csv"))
    except Exception as e:
        logger.error(f"Error loading processed SIRV data: {e}")
        data['sirv_processed'] = None
    
    try:
        if results['sc_original']['success']:
            data['sc_original'] = pd.read_csv(os.path.join(sc_dir, "bamslam_data.csv"))
            data['sc_original_stats'] = pd.read_csv(os.path.join(sc_dir, "bamslam_stats.csv"))
            data['sc_original_tx_stats'] = pd.read_csv(os.path.join(sc_dir, "bamslam_transcript_stats.csv"))
    except Exception as e:
        logger.error(f"Error loading scRNA-seq data: {e}")
        data['sc_original'] = None
    
    return data

def generate_comparative_plots(data, output_dir):
    """
    Generate comparative plots for the three datasets.
    
    Args:
        data: Dictionary with loaded DataFrames
        output_dir: Output directory
    """
    logger = logging.getLogger()
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "comparative_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set common style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    dataset_labels = ['Original SIRV', 'Processed SIRV', 'scRNA-seq']
    
    # Check which datasets are available
    available_datasets = []
    if data.get('sirv_original') is not None:
        available_datasets.append('sirv_original')
    if data.get('sirv_processed') is not None:
        available_datasets.append('sirv_processed')
    if data.get('sc_original') is not None:
        available_datasets.append('sc_original')
    
    if len(available_datasets) < 2:
        logger.warning("Not enough datasets available for comparative plots")
        return
    
    # 1. Coverage Fraction Comparison
    try:
        logger.info("Generating comparative coverage fraction plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, dataset in enumerate(available_datasets):
            # Get primary alignments
            primary = data[dataset].sort_values(['read_id', 'mapq', 'aligned_fraction'], ascending=[True, False, False])
            primary = primary.groupby('read_id').first().reset_index()
            
            # Plot KDE of coverage fractions
            sns.kdeplot(
                primary['read_coverage'], 
                ax=ax, 
                label=dataset_labels[i],
                color=colors[i]
            )
        
        # Add vertical line at 0.95 (full-length cutoff)
        ax.axvline(x=0.95, color='black', linestyle='dashed', linewidth=0.5, label='Full-length cutoff (0.95)')
        
        # Add labels and legend
        ax.set_xlabel("Coverage fraction")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.legend(title="Dataset")
        ax.set_title("Comparative Coverage Fraction Distribution")
        
        # Save plot
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "comparative_coverage_fraction.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparative coverage fraction plot to {plot_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating comparative coverage fraction plot: {e}")
    
    # 2. Coverage vs Length Comparison
    try:
        logger.info("Generating comparative coverage vs length plot")
        fig, axes = plt.subplots(1, len(available_datasets), figsize=(15, 5), sharey=True)
        
        for i, dataset in enumerate(available_datasets):
            # Get primary alignments
            primary = data[dataset].sort_values(['read_id', 'mapq', 'aligned_fraction'], ascending=[True, False, False])
            primary = primary.groupby('read_id').first().reset_index()
            
            # Create hexbin plot
            ax = axes[i] if len(available_datasets) > 1 else axes
            hb = ax.hexbin(
                primary['transcript_length'], 
                primary['read_coverage'], 
                gridsize=50, 
                cmap='viridis',
                bins='log',
                mincnt=1
            )
            
            # Add labels
            ax.set_xlabel("Transcript length (nt)")
            if i == 0:
                ax.set_ylabel("Coverage fraction")
            ax.set_title(dataset_labels[i])
            ax.set_xlim(0, min(15000, primary['transcript_length'].max() * 1.1))
            ax.set_ylim(0, 1)
            
            # Add colorbar
            if i == len(available_datasets) - 1:
                cbar = plt.colorbar(hb, ax=ax)
                cbar.set_label('log(count)')
        
        # Add overall title
        plt.suptitle("Coverage vs Transcript Length Comparison")
        
        # Save plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(plots_dir, "comparative_coverage_vs_length.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparative coverage vs length plot to {plot_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating comparative coverage vs length plot: {e}")
    
    # 3. Read Accuracy Comparison
    try:
        logger.info("Generating comparative read accuracy plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, dataset in enumerate(available_datasets):
            # Get primary alignments
            primary = data[dataset].sort_values(['read_id', 'mapq', 'aligned_fraction'], ascending=[True, False, False])
            primary = primary.groupby('read_id').first().reset_index()
            
            # Drop rows with NaN accuracy
            primary = primary.dropna(subset=['read_accuracy'])
            
            # Plot KDE of read accuracies
            sns.kdeplot(
                primary['read_accuracy'], 
                ax=ax, 
                label=dataset_labels[i],
                color=colors[i]
            )
        
        # Add labels and legend
        ax.set_xlabel("Read accuracy")
        ax.set_ylabel("Density")
        ax.set_xlim(0.5, 1)
        ax.legend(title="Dataset")
        ax.set_title("Comparative Read Accuracy Distribution")
        
        # Save plot
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "comparative_read_accuracy.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparative read accuracy plot to {plot_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating comparative read accuracy plot: {e}")
    
    # 4. Transcript Length Distribution Comparison
    try:
        logger.info("Generating comparative transcript length distribution plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, dataset in enumerate(available_datasets):
            # Get unique transcripts
            transcripts = data[dataset][['transcript_id', 'transcript_length']].drop_duplicates()
            
            # Plot KDE of transcript lengths
            sns.kdeplot(
                transcripts['transcript_length'], 
                ax=ax, 
                label=dataset_labels[i],
                color=colors[i]
            )
        
        # Add labels and legend
        ax.set_xlabel("Transcript length (nt)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 10000)
        ax.legend(title="Dataset")
        ax.set_title("Comparative Transcript Length Distribution")
        
        # Save plot
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "comparative_transcript_length.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparative transcript length plot to {plot_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating comparative transcript length plot: {e}")

def generate_summary_report(data, results, output_dir, args):
    """
    Generate a summary HTML report.
    
    Args:
        data: Dictionary with loaded DataFrames
        results: Dictionary with analysis results
        output_dir: Output directory
        args: Command-line arguments
    """
    logger = logging.getLogger()
    
    logger.info("Generating summary report")
    
    # Create summary statistics table
    stats_table = pd.DataFrame()
    
    # Metrics to include
    metrics = [
        'total_reads',
        'full_length_reads',
        'full_length_percentage',
        'median_coverage',
        'median_aligned_length',
        'median_accuracy',
        'unique_transcripts',
        'median_transcript_coverage',
        'median_transcript_length'
    ]
    
    # Rename metrics for better readability
    metric_names = {
        'total_reads': 'Total Reads',
        'full_length_reads': 'Full-Length Reads',
        'full_length_percentage': 'Full-Length Percentage (%)',
        'median_coverage': 'Median Coverage Fraction',
        'median_aligned_length': 'Median Aligned Length (bp)',
        'median_accuracy': 'Median Read Accuracy (%)',
        'unique_transcripts': 'Unique Transcripts',
        'median_transcript_coverage': 'Median Transcript Coverage',
        'median_transcript_length': 'Median Transcript Length (bp)'
    }
    
    # Check which datasets are available
    dataset_stats = {}
    if data.get('sirv_original_stats') is not None:
        stats_dict = dict(zip(data['sirv_original_stats']['metric'], data['sirv_original_stats']['value']))
        dataset_stats['Original SIRV'] = stats_dict
    
    if data.get('sirv_processed_stats') is not None:
        stats_dict = dict(zip(data['sirv_processed_stats']['metric'], data['sirv_processed_stats']['value']))
        dataset_stats['Processed SIRV'] = stats_dict
    
    if data.get('sc_original_stats') is not None:
        stats_dict = dict(zip(data['sc_original_stats']['metric'], data['sc_original_stats']['value']))
        dataset_stats['scRNA-seq'] = stats_dict
    
    # Build table
    rows = []
    for metric in metrics:
        row = {'Metric': metric_names.get(metric, metric)}
        for dataset, stats in dataset_stats.items():
            value = stats.get(metric, 'N/A')
            
            # Format numbers
            if isinstance(value, (int, np.integer)):
                row[dataset] = f"{value:,}"
            elif isinstance(value, (float, np.floating)):
                if metric in ['full_length_percentage', 'median_accuracy']:
                    row[dataset] = f"{value:.2f}%"
                elif metric in ['median_coverage', 'median_transcript_coverage']:
                    row[dataset] = f"{value:.4f}"
                else:
                    row[dataset] = f"{value:.2f}"
            else:
                row[dataset] = str(value)
        
        rows.append(row)
    
    stats_table = pd.DataFrame(rows)
    
    # Create HTML report
    report_path = os.path.join(output_dir, "comparative_report.html")
    
    # HTML header with CSS
    html_header = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SIRV Integration Comparative Report</title>
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
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .highlight {{
                background-color: #e6f7ff;
            }}
            .plot-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
                margin: 20px 0;
            }}
            .plot {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .summary {{
                background-color: #f9f9f9;
                border-left: 4px solid #2c3e50;
                padding: 15px;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                text-align: center;
                font-size: 0.9em;
                color: #777;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SIRV Integration Comparative Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Analysis Summary</h2>
                <p>This report compares BamSlam metrics across three datasets to evaluate how well the SIRV coverage model performed:</p>
                <ul>
                    <li><strong>Original SIRV:</strong> The raw SIRV alignment data before processing</li>
                    <li><strong>Processed SIRV:</strong> SIRV reads after being processed by the coverage model</li>
                    <li><strong>scRNA-seq:</strong> The original scRNA-seq data that the model aims to match</li>
                </ul>
            </div>
    """
    
    # Add comparative metrics table
    html_metrics = f"""
            <h2>Comparative Metrics</h2>
            <p>The table below shows key metrics across all datasets:</p>
            {stats_table.to_html(index=False, classes='table')}
    """
    
    # Add coverage model effectiveness section
    # Calculate similarity scores between processed SIRV and scRNA-seq
    similarity_html = ""
    if data.get('sirv_processed') is not None and data.get('sc_original') is not None:
        try:
            # Get primary alignments
            processed_primary = data['sirv_processed'].sort_values(
                ['read_id', 'mapq', 'aligned_fraction'], 
                ascending=[True, False, False]
            ).groupby('read_id').first().reset_index()
            
            sc_primary = data['sc_original'].sort_values(
                ['read_id', 'mapq', 'aligned_fraction'], 
                ascending=[True, False, False]
            ).groupby('read_id').first().reset_index()
            
            # Calculate median values for coverage
            proc_median_coverage = processed_primary['read_coverage'].median()
            sc_median_coverage = sc_primary['read_coverage'].median()
            
            # Calculate percentage difference
            coverage_diff = abs(proc_median_coverage - sc_median_coverage) / sc_median_coverage * 100
            
            # Determine effectiveness
            if coverage_diff < 10:
                effectiveness = "Excellent"
                desc = "The coverage model has successfully transformed SIRV reads to closely match the scRNA-seq coverage profile."
            elif coverage_diff < 20:
                effectiveness = "Good"
                desc = "The coverage model has performed well in transforming SIRV reads to match the scRNA-seq coverage profile."
            elif coverage_diff < 30:
                effectiveness = "Moderate"
                desc = "The coverage model has moderately transformed SIRV reads to match the scRNA-seq coverage profile."
            else:
                effectiveness = "Poor"
                desc = "The coverage model has not effectively transformed SIRV reads to match the scRNA-seq coverage profile."
            
            similarity_html = f"""
            <h2>Coverage Model Effectiveness</h2>
            <div class="summary">
                <h3>Model Evaluation: {effectiveness}</h3>
                <p>{desc}</p>
                <ul>
                    <li>Median coverage in processed SIRV: {proc_median_coverage:.4f}</li>
                    <li>Median coverage in scRNA-seq: {sc_median_coverage:.4f}</li>
                    <li>Percentage difference: {coverage_diff:.2f}%</li>
                </ul>
            </div>
            """
        except Exception as e:
            logger.error(f"Error calculating similarity scores: {e}")
            similarity_html = """
            <h2>Coverage Model Effectiveness</h2>
            <p>Unable to calculate similarity scores due to an error.</p>
            """
    
    # Add comparative plots section
    plots_dir = os.path.join(output_dir, "comparative_plots")
    plots_html = """
            <h2>Comparative Visualizations</h2>
            <p>The plots below show side-by-side comparisons of key metrics:</p>
            
            <h3>Coverage Fraction Distribution</h3>
            <div class="plot-container">
                <img class="plot" src="comparative_plots/comparative_coverage_fraction.png" alt="Coverage Fraction Comparison">
            </div>
            <p>This plot shows how read coverage is distributed across the different datasets. If the coverage model is working effectively, the processed SIRV line (orange) should move closer to the scRNA-seq line (green) compared to the original SIRV line (blue).</p>
            
            <h3>Coverage vs Transcript Length</h3>
            <div class="plot-container">
                <img class="plot" src="comparative_plots/comparative_coverage_vs_length.png" alt="Coverage vs Length Comparison">
            </div>
            <p>This plot shows how coverage varies with transcript length. Effective coverage modeling should make the processed SIRV pattern more similar to the scRNA-seq pattern.</p>
            
            <h3>Read Accuracy Distribution</h3>
            <div class="plot-container">
                <img class="plot" src="comparative_plots/comparative_read_accuracy.png" alt="Read Accuracy Comparison">
            </div>
            <p>This plot shows the distribution of read accuracies across datasets. The coverage model may affect read accuracy.</p>
            
            <h3>Transcript Length Distribution</h3>
            <div class="plot-container">
                <img class="plot" src="comparative_plots/comparative_transcript_length.png" alt="Transcript Length Comparison">
            </div>
            <p>This plot shows the distribution of transcript lengths across datasets. Similar distributions indicate good integration of SIRV transcripts.</p>
    """
    
    # HTML footer
    html_footer = """
            <div class="footer">
                <p>Generated by SIRV Integration Pipeline Comparative Analysis</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Combine all HTML sections
    html_content = html_header + html_metrics + similarity_html + plots_html + html_footer
    
    # Write HTML report
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Saved summary report to {report_path}")

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    log_file = os.path.join(args.output_dir, "comparative_analysis.log")
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(log_file, console_level=log_level, file_level=logging.DEBUG)
    
    logger.info("Starting Comparative BamSlam Analysis")
    logger.info(f"Original SIRV BAM: {args.sirv_bam}")
    logger.info(f"Integrated FASTQ: {args.integrated_fastq}")
    logger.info(f"scRNA-seq BAM: {args.sc_bam}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Run analysis on all datasets
        results = analyze_all_datasets(args)
        
        # Load analysis data
        data = load_analysis_data(results, args.output_dir)
        
        # Generate comparative plots
        generate_comparative_plots(data, args.output_dir)
        
        # Generate summary report
        generate_summary_report(data, results, args.output_dir, args)
        
        logger.info("Comparative analysis completed successfully")
        logger.info(f"Summary report: {os.path.join(args.output_dir, 'comparative_report.html')}")
        
        return 0
    except Exception as e:
        logger.error(f"Error during comparative analysis: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 