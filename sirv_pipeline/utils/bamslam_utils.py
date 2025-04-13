"""
Utility functions for comparative BamSlam analysis.

This module provides helper functions for processing FASTQ and BAM files
for the comparative BamSlam analysis pipeline.
"""

import os
import sys
import logging
import re
import gzip
import pandas as pd
import subprocess
import tempfile
import traceback
from pathlib import Path
import pysam

def setup_bamslam_logger(log_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """Set up logger for the BamSlam analysis script."""
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
        
        # If no reads were extracted, this is likely an error
        if count == 0:
            logger.warning("No SIRV reads found using pattern matching")
            if sirv_extracted_fastq and os.path.exists(sirv_extracted_fastq):
                logger.info(f"Using pre-extracted SIRV reads: {sirv_extracted_fastq}")
                # Just copy the extracted FASTQ
                import shutil
                shutil.copy(sirv_extracted_fastq, filtered_fastq)
                logger.info(f"Copied pre-extracted SIRV reads to {filtered_fastq}")
                return filtered_fastq
    except Exception as e:
        logger.error(f"Error extracting SIRV reads: {str(e)}")
        logger.error(traceback.format_exc())
        
        if sirv_extracted_fastq and os.path.exists(sirv_extracted_fastq):
            logger.info(f"Using pre-extracted SIRV reads: {sirv_extracted_fastq}")
            # Just copy the extracted FASTQ
            import shutil
            shutil.copy(sirv_extracted_fastq, filtered_fastq)
            logger.info(f"Copied pre-extracted SIRV reads to {filtered_fastq}")
            return filtered_fastq
    
    return filtered_fastq

def map_fastq_to_reference(fastq_file, reference_file, output_dir, threads=4):
    """
    Map reads from a FASTQ file to a reference genome.
    
    Args:
        fastq_file: Path to FASTQ file
        reference_file: Path to reference FASTA file
        output_dir: Output directory
        threads: Number of threads for mapping
        
    Returns:
        Path to BAM file
    """
    logger = logging.getLogger()
    logger.info(f"Mapping FASTQ to reference: {fastq_file} -> {reference_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Output BAM path
    output_bam = os.path.join(output_dir, "mapped_reads.bam")
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Run minimap2 to map reads
        logger.info("Running minimap2 for read mapping")
        sam_file = os.path.join(temp_dir, "mapped_reads.sam")
        
        # Build minimap2 command
        minimap2_cmd = [
            "minimap2",
            "-ax", "map-ont",  # Oxford Nanopore preset
            "-t", str(threads),
            reference_file,
            fastq_file
        ]
        
        # Run minimap2
        with open(sam_file, 'w') as sam_out:
            logger.info(f"Running: {' '.join(minimap2_cmd)}")
            subprocess.run(minimap2_cmd, stdout=sam_out, check=True)
        
        # Convert SAM to BAM
        logger.info("Converting SAM to BAM")
        
        # Build samtools view command
        view_cmd = [
            "samtools", "view",
            "-bS",
            "-o", output_bam,
            sam_file
        ]
        
        # Run samtools view
        logger.info(f"Running: {' '.join(view_cmd)}")
        subprocess.run(view_cmd, check=True)
        
        # Sort BAM file
        logger.info("Sorting BAM file")
        sorted_bam = os.path.join(temp_dir, "sorted.bam")
        
        # Build samtools sort command
        sort_cmd = [
            "samtools", "sort",
            "-o", sorted_bam,
            output_bam
        ]
        
        # Run samtools sort
        logger.info(f"Running: {' '.join(sort_cmd)}")
        subprocess.run(sort_cmd, check=True)
        
        # Replace output BAM with sorted BAM
        import shutil
        shutil.move(sorted_bam, output_bam)
        
        # Index BAM file
        logger.info("Indexing BAM file")
        
        # Build samtools index command
        index_cmd = [
            "samtools", "index",
            output_bam
        ]
        
        # Run samtools index
        logger.info(f"Running: {' '.join(index_cmd)}")
        subprocess.run(index_cmd, check=True)
        
        # Clean up temporary directory
        logger.info(f"Removing temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        
        logger.info(f"Mapping completed: {output_bam}")
        return output_bam
    
    except Exception as e:
        logger.error(f"Error mapping FASTQ to reference: {str(e)}")
        logger.error(traceback.format_exc())
        return None 