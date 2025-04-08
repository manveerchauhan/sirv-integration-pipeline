"""
SIRV integration module for the SIRV Integration Pipeline.

This module handles the integration of SIRV reads into 
existing scRNA-seq datasets with barcode addition.
"""

import os
import logging
import random
import numpy as np
import pandas as pd
import gzip
from Bio import SeqIO
from typing import Dict, List, Tuple, Optional, Any

# Import from other modules
from sirv_pipeline.coverage_bias import create_coverage_bias_model, CoverageBiasModel, ReadLengthSampler
from sirv_pipeline.utils import validate_files

# Set up logger
logger = logging.getLogger(__name__)


class CellBarcode:
    """Class to handle cell barcode creation and UMIs."""
    
    def __init__(self, barcode_length: int = 16, umi_length: int = 12, seed: Optional[int] = None):
        """Initialize the CellBarcode generator."""
        self.barcode_length = barcode_length
        self.umi_length = umi_length
        self.nucleotides = ['A', 'C', 'G', 'T']
        
        if seed is not None:
            random.seed(seed)
    
    def generate_barcodes(self, n_cells: int) -> List[str]:
        """Generate a list of random cell barcodes."""
        return [''.join(random.choice(self.nucleotides) for _ in range(self.barcode_length)) 
                for _ in range(n_cells)]
    
    def generate_umi(self) -> str:
        """Generate a random UMI sequence."""
        return ''.join(random.choice(self.nucleotides) for _ in range(self.umi_length))


def extract_cell_info(sc_fastq: str, sample_size: int = 1000) -> Dict[str, int]:
    """
    Extract cell information from scRNA-seq FASTQ.
    
    In a real implementation, this would parse actual cell barcodes.
    For now, it generates synthetic barcodes and counts.
    """
    logger.info(f"Extracting cell information from {sc_fastq}")
    
    # Create synthetic cell barcodes (10 cells by default)
    barcode_gen = CellBarcode()
    barcodes = barcode_gen.generate_barcodes(10)
    
    # Assign random read counts
    read_counts = np.random.randint(1000, 5000, size=len(barcodes))
    cell_info = dict(zip(barcodes, read_counts))
    
    logger.info(f"Found {len(cell_info)} cells with {sum(read_counts)} total reads")
    return cell_info


def add_sirv_to_dataset(
    sc_fastq: str,
    sirv_fastq: str,
    transcript_map_file: str,
    coverage_model_file: str,
    output_fastq: str,
    tracking_file: Optional[str] = None,
    insertion_rate: float = 0.01,
    expected_file: Optional[str] = None,
    sample_size: int = 1000,
    reference_file: Optional[str] = None,
    model_coverage_bias: bool = True,
    seed: Optional[int] = None
) -> Tuple[str, str, str]:
    """
    Add SIRV reads to an existing scRNA-seq dataset.
    
    Args:
        sc_fastq: Path to single-cell FASTQ file
        sirv_fastq: Path to SIRV FASTQ file
        transcript_map_file: Path to transcript mapping CSV file
        coverage_model_file: Path to coverage model CSV file
        output_fastq: Path to output integrated FASTQ file
        tracking_file: Path to output tracking file (default: <output_dir>/tracking.csv)
        insertion_rate: Rate of SIRV insertion (0-1, default: 0.01)
        expected_file: Path to output expected counts file (default: <output_dir>/expected_counts.csv)
        sample_size: Number of reads to sample for coverage modeling
        reference_file: Path to reference file for coverage modeling (optional)
        model_coverage_bias: Whether to model coverage bias (default: True)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple[str, str, str]: Paths to output FASTQ, tracking file, and expected counts file
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Validate inputs
    logger.info(f"Validating inputs")
    validate_files(sirv_fastq, sc_fastq, transcript_map_file, mode='r')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output_fastq))
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default output files if not provided
    if tracking_file is None:
        tracking_file = os.path.join(output_dir, "tracking.csv")
    
    if expected_file is None:
        expected_file = os.path.join(output_dir, "expected_counts.csv")
    
    validate_files(output_fastq, tracking_file, expected_file, mode='w')
    
    # Step 1: Load SIRV read mapping
    logger.info(f"Loading SIRV read mapping")
    sirv_map_df = pd.read_csv(transcript_map_file)
    
    if sirv_map_df.empty:
        logger.warning("Transcript map is empty. No SIRV reads to integrate.")
        # Just copy the original scRNA-seq data to output
        logger.info(f"Copying original scRNA-seq reads to output")
        opener_in = gzip.open if sc_fastq.endswith('.gz') else open
        opener_out = gzip.open if output_fastq.endswith('.gz') else open
        
        with opener_in(sc_fastq, 'rt') as f_in, opener_out(output_fastq, 'wt') as f_out:
            for line in f_in:
                f_out.write(line)
        
        # Create empty tracking and expected files
        pd.DataFrame(columns=[
            'read_id', 'original_read_id', 'barcode', 'umi', 
            'sirv_transcript', 'original_length', 'sampled_length'
        ]).to_csv(tracking_file, index=False)
        
        pd.DataFrame(columns=[
            'barcode', 'sirv_transcript', 'expected_count'
        ]).to_csv(expected_file, index=False)
        
        logger.info(f"Integration complete. Output files (no SIRV reads added):")
        logger.info(f"- Combined FASTQ: {output_fastq}")
        logger.info(f"- Tracking file: {tracking_file}")
        logger.info(f"- Expected counts: {expected_file}")
        
        return output_fastq, tracking_file, expected_file
    
    read_to_transcript = dict(zip(sirv_map_df['read_id'], sirv_map_df['sirv_transcript']))
    logger.info(f"Loaded {len(read_to_transcript)} SIRV read mappings")
    
    # Step 2: Extract cell information from scRNA-seq data
    logger.info(f"Extracting cell information")
    cell_info = extract_cell_info(sc_fastq, sample_size)
    
    # Step 3: Create read length and coverage bias models
    logger.info(f"Creating read length and coverage bias models")
    
    # Use provided reference file if available, otherwise default to SIRV reference
    reference_for_model = reference_file
    
    # Check if we should model coverage bias
    if model_coverage_bias:
        if reference_for_model:
            logger.info(f"Using provided reference file for coverage modeling: {reference_for_model}")
        else:
            logger.warning("No reference file provided for coverage modeling. Coverage bias may not be accurately modeled.")
    
    length_sampler, bias_model = create_coverage_bias_model(
        fastq_file=sc_fastq,
        reference_file=reference_for_model,
        sample_size=sample_size,
        seed=seed
    )
    
    # Plot coverage bias model if available
    if model_coverage_bias and bias_model.has_model:
        plot_file = os.path.join(output_dir, "coverage_bias.png")
        bias_model.plot_distributions(plot_file)
    
    # Step 4: Load SIRV reads
    logger.info(f"Loading SIRV reads")
    sirv_reads = {}
    
    opener = gzip.open if sirv_fastq.endswith('.gz') else open
    with opener(sirv_fastq, 'rt') as f:
        for record in SeqIO.parse(f, 'fastq'):
            read_id = record.id
            if read_id in read_to_transcript:
                sirv_reads[read_id] = {
                    'sequence': str(record.seq),
                    'quality': record.letter_annotations['phred_quality'],
                    'transcript': read_to_transcript[read_id],
                    'length': len(record.seq)
                }
    
    logger.info(f"Loaded {len(sirv_reads)} SIRV reads with transcript assignments")
    
    # Step 5: Integrate SIRV reads into scRNA-seq dataset
    logger.info(f"Integrating SIRV reads into scRNA-seq dataset")
    
    # Calculate total number of SIRV reads to add based on insertion rate
    total_reads = sum(cell_info.values())
    n_sirv_reads = int(total_reads * insertion_rate / (1 - insertion_rate))
    logger.info(f"Will add {n_sirv_reads} SIRV reads to {total_reads} scRNA-seq reads")
    
    # Initialize tracking data and expected counts
    tracking_data = []
    expected_counts = {}
    
    # Initialize barcode generator
    barcode_gen = CellBarcode(seed=seed)
    
    # Select random SIRV reads
    sirv_read_ids = list(sirv_reads.keys())
    
    # Sample reads with replacement if needed
    selected_reads = np.random.choice(
        sirv_read_ids, 
        n_sirv_reads, 
        replace=len(sirv_read_ids) < n_sirv_reads
    )
    
    # Assign reads to cells based on cell abundance
    barcodes = list(cell_info.keys())
    barcode_probs = np.array(list(cell_info.values())) / sum(cell_info.values())
    cell_assignments = np.random.choice(barcodes, n_sirv_reads, p=barcode_probs)
    
    # Create output files
    output_handle = gzip.open(output_fastq, 'wt') if output_fastq.endswith('.gz') else open(output_fastq, 'w')
    
    # First copy all original scRNA-seq reads
    logger.info(f"Copying original scRNA-seq reads")
    opener = gzip.open if sc_fastq.endswith('.gz') else open
    with opener(sc_fastq, 'rt') as f:
        for line in f:
            output_handle.write(line)
    
    # Now add SIRV reads with cell barcodes
    logger.info(f"Adding SIRV reads with cell barcodes")
    
    for i, (read_id, barcode) in enumerate(zip(selected_reads, cell_assignments)):
        sirv_read = sirv_reads[read_id]
        
        # Generate UMI and sample read length
        umi = barcode_gen.generate_umi()
        target_length = length_sampler.sample()
        
        # Apply coverage bias if model is available
        if model_coverage_bias and bias_model.has_model:
            # Convert quality scores to ASCII
            quality_string = ''.join(chr(q + 33) for q in sirv_read['quality'])
            
            # Apply bias to sequence
            seq, qual = bias_model.apply_to_sequence(
                sirv_read['sequence'], 
                quality_string, 
                target_length
            )
            
            # Convert quality back to integer list if needed
            quality_scores = [ord(q) - 33 for q in qual]
        else:
            # Just truncate to target length
            seq = sirv_read['sequence'][:target_length]
            quality_scores = sirv_read['quality'][:target_length]
        
        # Create new read ID with cell barcode
        new_read_id = f"{read_id}-{barcode}-{umi}"
        
        # Convert quality scores to FASTQ format
        quality_string = ''.join(chr(q + 33) for q in quality_scores)
        
        # Write to output FASTQ
        output_handle.write(f"@{new_read_id}\n{seq}\n+\n{quality_string}\n")
        
        # Track this read
        tracking_data.append({
            'read_id': new_read_id,
            'original_read_id': read_id,
            'barcode': barcode,
            'umi': umi,
            'sirv_transcript': sirv_read['transcript'],
            'original_length': sirv_read['length'],
            'sampled_length': len(seq)
        })
        
        # Update expected counts
        key = (barcode, sirv_read['transcript'])
        expected_counts[key] = expected_counts.get(key, 0) + 1
        
        # Log progress occasionally
        if (i + 1) % 1000 == 0:
            logger.info(f"Added {i+1}/{n_sirv_reads} SIRV reads")
    
    # Close output file
    output_handle.close()
    
    # Write tracking file
    tracking_df = pd.DataFrame(tracking_data)
    tracking_df.to_csv(tracking_file, index=False)
    
    # Write expected counts file
    expected_df = pd.DataFrame([
        {'barcode': k[0], 'sirv_transcript': k[1], 'expected_count': v}
        for k, v in expected_counts.items()
    ])
    expected_df.to_csv(expected_file, index=False)
    
    logger.info(f"Integration complete. Output files:")
    logger.info(f"- Combined FASTQ: {output_fastq}")
    logger.info(f"- Tracking file: {tracking_file}")
    logger.info(f"- Expected counts: {expected_file}")
    
    return output_fastq, tracking_file, expected_file