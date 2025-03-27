"""
SIRV read mapping module for the SIRV Integration Pipeline.

This module handles mapping SIRV reads to reference genomes and 
identifying transcript of origin for each read.
"""

import os
import logging
import tempfile
import subprocess
import pandas as pd
import pysam
from typing import Dict, List, Tuple, Optional, Union

# Set up logger
logger = logging.getLogger(__name__)


def map_sirv_reads(
    sirv_fastq: str,
    sirv_reference: str,
    sirv_gtf: str,
    output_csv: str,
    threads: int = 4,
    min_overlap: float = 0.5,
    keep_temp: bool = False
) -> str:
    """
    Map SIRV reads to reference using minimap2 with ONT settings.
    
    Args:
        sirv_fastq: Path to SIRV FASTQ file
        sirv_reference: Path to SIRV reference genome
        sirv_gtf: Path to SIRV annotation GTF
        output_csv: Path to output CSV file for transcript mappings
        threads: Number of threads for parallel processing
        min_overlap: Minimum overlap fraction required for transcript assignment
        keep_temp: Keep temporary files for debugging
        
    Returns:
        str: Path to the output CSV file with read-to-transcript mappings
        
    Raises:
        FileNotFoundError: If input files do not exist
        subprocess.CalledProcessError: If alignment process fails
    """
    # Validate input files
    for input_file, description in [
        (sirv_fastq, "SIRV FASTQ"), 
        (sirv_reference, "SIRV reference"), 
        (sirv_gtf, "SIRV GTF")
    ]:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"{description} file not found: {input_file}")
    
    logger.info(f"Mapping SIRV reads to reference using minimap2...")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    try:
        prefix = os.path.basename(sirv_fastq).split('.')[0]
        
        # Define output files
        bam_file = os.path.join(temp_dir, f"{prefix}.bam")
        sam_file = os.path.join(temp_dir, f"{prefix}.sam")
        
        # Run minimap2 with ONT preset
        logger.info(f"Aligning reads with minimap2...")
        cmd = [
            "minimap2", "-ax", "map-ont",     # ONT preset
            "-k", "14",                       # K-mer size for ONT
            "--secondary=no",                 # Don't output secondary alignments
            "-t", str(threads),
            sirv_reference, sirv_fastq
        ]
        
        logger.debug(f"Running command: {' '.join(cmd)}")
        with open(sam_file, 'w') as f:
            subprocess.run(cmd, stdout=f, check=True)
        
        # Convert SAM to BAM and sort
        logger.info(f"Converting SAM to sorted BAM...")
        subprocess.run(["samtools", "sort", "-o", bam_file, sam_file], check=True)
        subprocess.run(["samtools", "index", bam_file], check=True)
        
        # Load transcript info from GTF
        transcripts = _parse_transcripts_from_gtf(sirv_gtf)
        logger.info(f"Loaded {len(transcripts)} transcripts from GTF")
        
        # Process BAM to identify transcripts
        mappings = _assign_transcripts_to_reads(bam_file, transcripts, min_overlap)
        
        # Save mappings to CSV
        df = pd.DataFrame(mappings)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        
        logger.info(f"Found {len(df)} SIRV reads with transcript assignments")
        logger.info(f"Unique SIRV transcripts identified: {df['sirv_transcript'].nunique()}")
        
        return output_csv
        
    finally:
        # Clean up temporary files unless keep_temp is True
        if not keep_temp:
            import shutil
            logger.debug(f"Removing temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        else:
            logger.info(f"Keeping temporary files in: {temp_dir}")


def _parse_transcripts_from_gtf(gtf_file: str) -> Dict[str, Dict]:
    """
    Parse transcript information from GTF file.
    
    Args:
        gtf_file: Path to GTF annotation file
        
    Returns:
        Dict[str, Dict]: Dictionary mapping transcript IDs to transcript information
    """
    transcripts = {}
    
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'transcript':
                continue
                
            # Extract transcript_id
            attr_str = fields[8]
            tx_id = None
            
            for attr in attr_str.split(';'):
                attr = attr.strip()
                if attr.startswith('transcript_id'):
                    # Handle both quote formats
                    if '"' in attr:
                        tx_id = attr.split('"')[1]
                    else:
                        tx_id = attr.split(' ')[1]
                    break
            
            if tx_id:
                transcripts[tx_id] = {
                    'chrom': fields[0],
                    'start': int(fields[3]),
                    'end': int(fields[4]),
                    'strand': fields[6]
                }
    
    return transcripts


def _assign_transcripts_to_reads(
    bam_file: str, 
    transcripts: Dict[str, Dict], 
    min_overlap: float = 0.5
) -> List[Dict]:
    """
    Process BAM file to assign reads to transcripts.
    
    Args:
        bam_file: Path to BAM file
        transcripts: Dictionary of transcript information
        min_overlap: Minimum overlap fraction required for assignment
        
    Returns:
        List[Dict]: List of dictionaries with read to transcript mappings
    """
    mappings = []
    
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for read in bam:
            if read.is_unmapped:
                continue
            
            chrom = read.reference_name
            start = read.reference_start
            end = read.reference_end or read.reference_start + read.query_length
            
            # Find best matching transcript
            best_tx = None
            best_overlap = 0
            
            for tx_id, tx in transcripts.items():
                if tx['chrom'] == chrom:
                    # Calculate overlap
                    overlap_start = max(start, tx['start'])
                    overlap_end = min(end, tx['end'])
                    overlap = max(0, overlap_end - overlap_start)
                    
                    # Calculate overlap fraction relative to read length
                    read_len = end - start
                    if read_len > 0:
                        overlap_fraction = overlap / read_len
                        
                        if overlap_fraction > best_overlap:
                            best_overlap = overlap_fraction
                            best_tx = tx_id
            
            # Only save mappings that meet the minimum overlap threshold
            if best_tx and best_overlap >= min_overlap:
                mappings.append({
                    'read_id': read.query_name,
                    'sirv_transcript': best_tx,
                    'overlap_fraction': best_overlap,
                    'read_length': read.query_length,
                    'alignment_length': end - start
                })
    
    return mappings


def get_transcript_statistics(mapping_csv: str) -> Dict[str, int]:
    """
    Get statistics about transcript assignments.
    
    Args:
        mapping_csv: Path to mapping CSV file
        
    Returns:
        Dict[str, int]: Dictionary with transcript statistics
    """
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(f"Mapping CSV file not found: {mapping_csv}")
    
    df = pd.read_csv(mapping_csv)
    
    stats = {
        'total_reads': len(df),
        'unique_transcripts': df['sirv_transcript'].nunique(),
        'reads_per_transcript': df.groupby('sirv_transcript').size().to_dict()
    }
    
    return stats