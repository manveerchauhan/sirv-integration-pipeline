"""
SIRV integration module for the SIRV Integration Pipeline.

This module handles the integration of SIRV reads into 
existing scRNA-seq datasets with barcode addition, 
read truncation, and output tracking.
"""

import os
import logging
import random
import numpy as np
import pandas as pd
import gzip
from Bio import SeqIO
from typing import Dict, List, Tuple, Optional, Any

# Set up logger
logger = logging.getLogger(__name__)


class CellBarcode:
    """
    Class to handle cell barcode creation and manipulation.
    """
    
    def __init__(self, barcode_length: int = 16, seed: Optional[int] = None):
        """
        Initialize the CellBarcode generator.
        
        Args:
            barcode_length: Length of generated barcodes
            seed: Random seed for reproducibility
        """
        self.barcode_length = barcode_length
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.nucleotides = ['A', 'C', 'G', 'T']
    
    def generate_barcodes(self, n_cells: int) -> List[str]:
        """
        Generate a list of random cell barcodes.
        
        Args:
            n_cells: Number of cell barcodes to generate
            
        Returns:
            List[str]: List of generated cell barcodes
        """
        barcodes = []
        for _ in range(n_cells):
            barcode = ''.join(random.choice(self.nucleotides) for _ in range(self.barcode_length))
            barcodes.append(barcode)
        return barcodes
    
    def generate_umi(self, length: int = 12) -> str:
        """
        Generate a random UMI (Unique Molecular Identifier).
        
        Args:
            length: Length of the UMI (default is 12 for 10x Genomics)
            
        Returns:
            str: Generated UMI sequence
        """
        return ''.join(random.choice(self.nucleotides) for _ in range(length))


class ReadLengthSampler:
    """
    Class to sample realistic read lengths from existing data.
    """
    
    def __init__(self, length_distribution: Optional[np.ndarray] = None, 
                 default_mean: int = 1000, default_std: int = 300, 
                 min_length: int = 100):
        """
        Initialize the read length sampler.
        
        Args:
            length_distribution: Optional array of read lengths to sample from
            default_mean: Mean read length if no distribution is provided
            default_std: Standard deviation if no distribution is provided
            min_length: Minimum allowed read length
        """
        self.length_distribution = length_distribution
        self.default_mean = default_mean
        self.default_std = default_std
        self.min_length = min_length
    
    def sample_from_fastq(self, fastq_file: str, 
                          sample_size: int = 1000) -> np.ndarray:
        """
        Sample read lengths from a FASTQ file.
        
        Args:
            fastq_file: Path to input FASTQ file
            sample_size: Number of reads to sample
            
        Returns:
            np.ndarray: Array of sampled read lengths
        """
        logger.info(f"Sampling read lengths from {fastq_file}...")
        
        lengths = []
        count = 0
        
        # Handle compressed files
        opener = gzip.open if fastq_file.endswith('.gz') else open
        
        try:
            with opener(fastq_file, 'rt') as f:
                for record in SeqIO.parse(f, 'fastq'):
                    lengths.append(len(record.seq))
                    count += 1
                    if count >= sample_size:
                        break
        except Exception as e:
            logger.warning(f"Error sampling read lengths: {e}")
            logger.warning("Using default length distribution instead")
            return self._generate_default_distribution(sample_size)
        
        if not lengths:
            logger.warning("No reads found, using default length distribution")
            return self._generate_default_distribution(sample_size)
        
        self.length_distribution = np.array(lengths)
        logger.info(f"Sampled {len(lengths)} read lengths (mean: {np.mean(lengths):.0f}, " 
                   f"median: {np.median(lengths):.0f}, min: {np.min(lengths)}, max: {np.max(lengths)})")
        
        return self.length_distribution
    
    def _generate_default_distribution(self, size: int = 1000) -> np.ndarray:
        """
        Generate a default read length distribution.
        
        Args:
            size: Number of values to generate
            
        Returns:
            np.ndarray: Array of generated read lengths
        """
        lengths = np.random.normal(self.default_mean, self.default_std, size=size)
        lengths = np.maximum(lengths, self.min_length).astype(int)
        self.length_distribution = lengths
        return lengths
    
    def sample(self) -> int:
        """
        Sample a read length.
        
        Returns:
            int: Sampled read length
        """
        if self.length_distribution is not None and len(self.length_distribution) > 0:
            return int(np.random.choice(self.length_distribution))
        else:
            return max(self.min_length, int(np.random.normal(self.default_mean, self.default_std)))


def extract_cell_info(sc_fastq: str) -> Dict[str, int]:
    """
    Extract cell information from a scRNA-seq FASTQ file.
    
    In real implementation, this would parse actual cell barcodes.
    For this example, we generate placeholder information.
    
    Args:
        sc_fastq: Path to scRNA-seq FASTQ file
        
    Returns:
        Dict[str, int]: Dictionary mapping cell barcodes to read counts
    """
    logger.info(f"Extracting cell information from {sc_fastq}...")
    
    # For simplicity - in a real implementation you'd extract actual barcodes
    # This example just creates placeholder barcodes and read counts
    
    # Create cell barcodes
    barcode_gen = CellBarcode()
    barcodes = barcode_gen.generate_barcodes(10)
    
    # Generate random read counts for each cell
    read_counts = np.random.randint(1000, 5000, size=len(barcodes))
    
    cell_info = dict(zip(barcodes, read_counts))
    
    logger.info(f"Found {len(cell_info)} cells with {sum(read_counts)} total reads")
    
    return cell_info


def add_sirv_to_dataset(
    sirv_fastq: str, 
    sc_fastq: str,
    sirv_map_csv: str,
    output_fastq: str,
    insertion_rate: float = 0.01,
    tracking_file: Optional[str] = None,
    expected_file: Optional[str] = None,
    umi_length: int = 12,
    sample_size: int = 1000,
    seed: Optional[int] = None
) -> Tuple[str, str, str]:
    """
    Add SIRV reads to existing scRNA-seq dataset with barcodes.
    
    Args:
        sirv_fastq: Path to SIRV FASTQ file
        sc_fastq: Path to scRNA-seq FASTQ file
        sirv_map_csv: Path to SIRV mapping CSV
        output_fastq: Path to output combined FASTQ
        insertion_rate: Proportion of SIRV reads to add
        tracking_file: Path to output tracking CSV
        expected_file: Path to output expected counts CSV
        umi_length: Length of generated UMIs
        sample_size: Number of reads to sample for length distribution
        seed: Random seed for reproducibility
        
    Returns:
        Tuple[str, str, str]: Paths to output FASTQ, tracking file, and expected counts file
        
    Raises:
        FileNotFoundError: If input files do not exist
        ValueError: If insertion_rate is outside valid range
    """
    # Validate inputs
    for input_file, description in [
        (sirv_fastq, "SIRV FASTQ"), 
        (sc_fastq, "scRNA-seq FASTQ"), 
        (sirv_map_csv, "SIRV mapping CSV")
    ]:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"{description} file not found: {input_file}")
    
    if insertion_rate <= 0 or insertion_rate > 0.5:
        raise ValueError(f"Insertion rate must be between 0 and 0.5, got {insertion_rate}")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    logger.info(f"Adding SIRV reads to dataset at rate {insertion_rate}...")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_fastq)), exist_ok=True)
    
    # Set default tracking and expected files if not provided
    if tracking_file is None:
        tracking_file = f"{os.path.splitext(output_fastq)[0]}_tracking.csv"
    
    if expected_file is None:
        expected_file = f"{os.path.splitext(output_fastq)[0]}_expected_counts.csv"
    
    # Load SIRV transcript mapping
    sirv_df = pd.read_csv(sirv_map_csv)
    read_to_transcript = dict(zip(sirv_df['read_id'], sirv_df['sirv_transcript']))
    
    # Load SIRV reads
    sirv_reads = _load_sirv_reads(sirv_fastq, read_to_transcript)
    
    # Get cell info
    cell_info = extract_cell_info(sc_fastq)
    
    # Sample read length distribution
    length_sampler = ReadLengthSampler()
    length_sampler.sample_from_fastq(sc_fastq, sample_size=sample_size)
    
    # Create cell barcode generator
    barcode_gen = CellBarcode(seed=seed)
    
    # Track SIRV additions
    tracking_data = []
    
    # Process reads
    with open(output_fastq, 'w') as outfile:
        # Add SIRV reads for each cell
        for barcode, read_count in cell_info.items():
            # Number of SIRVs to add
            num_sirvs = int(read_count * insertion_rate)
            
            if num_sirvs <= 0:
                logger.debug(f"Skipping cell {barcode}: insertion rate too low")
                continue
            
            # Available SIRV reads
            sirv_ids = list(sirv_reads.keys())
            
            # Sample reads
            if num_sirvs > len(sirv_ids):
                sampled_ids = random.choices(sirv_ids, k=num_sirvs)
            else:
                sampled_ids = random.sample(sirv_ids, num_sirvs)
            
            # Add SIRV reads with cell barcode
            for read_id in sampled_ids:
                sirv = sirv_reads[read_id]
                
                # Generate UMI
                umi = barcode_gen.generate_umi(umi_length)
                
                # Truncate to realistic read length
                target_len = min(length_sampler.sample(), len(sirv['seq']))
                seq = sirv['seq'][:target_len]
                qual = sirv['qual'][:target_len]
                
                # New read ID with tracking info
                new_id = f"{read_id}-{barcode}-{umi}"
                
                # Write to output
                outfile.write(f"@{new_id}\n{seq}\n+\n{qual}\n")
                
                # Track addition
                tracking_data.append({
                    'read_id': new_id,
                    'original_read_id': read_id,
                    'barcode': barcode,
                    'umi': umi,
                    'sirv_transcript': sirv['transcript']
                })
        
        # Copy original scRNA-seq reads
        _copy_fastq_contents(sc_fastq, outfile)
    
    # Save tracking information
    pd.DataFrame(tracking_data).to_csv(tracking_file, index=False)
    
    # Create expected counts
    expected_counts = (pd.DataFrame(tracking_data)
                       .groupby(['barcode', 'sirv_transcript'])
                       .size()
                       .reset_index(name='expected_count'))
    expected_counts.to_csv(expected_file, index=False)
    
    logger.info(f"Added {len(tracking_data)} SIRV reads to {len(cell_info)} cells")
    logger.info(f"Tracking info saved to: {tracking_file}")
    logger.info(f"Expected counts saved to: {expected_file}")
    
    return output_fastq, tracking_file, expected_file


def _load_sirv_reads(sirv_fastq: str, read_to_transcript: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Load SIRV reads from a FASTQ file.
    
    Args:
        sirv_fastq: Path to SIRV FASTQ file
        read_to_transcript: Dictionary mapping read IDs to transcript IDs
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of SIRV reads
    """
    logger.info(f"Loading SIRV reads from {sirv_fastq}...")
    
    sirv_reads = {}
    opener = gzip.open if sirv_fastq.endswith('.gz') else open
    
    try:
        with opener(sirv_fastq, 'rt') as f:
            for record in SeqIO.parse(f, 'fastq'):
                if record.id in read_to_transcript:
                    sirv_reads[record.id] = {
                        'seq': str(record.seq),
                        'qual': ''.join(chr(q+33) for q in record.letter_annotations['phred_quality']),
                        'transcript': read_to_transcript[record.id]
                    }
    except Exception as e:
        logger.error(f"Error loading SIRV reads: {e}")
        raise
    
    logger.info(f"Loaded {len(sirv_reads)} SIRV reads with transcript assignments")
    
    return sirv_reads


def _copy_fastq_contents(input_fastq: str, output_file_handle) -> int:
    """
    Copy contents from an input FASTQ file to an output file handle.
    
    Args:
        input_fastq: Path to input FASTQ file
        output_file_handle: File handle for output
        
    Returns:
        int: Number of reads copied
    """
    logger.info(f"Copying original reads from {input_fastq}...")
    
    read_count = 0
    opener = gzip.open if input_fastq.endswith('.gz') else open
    
    try:
        with opener(input_fastq, 'rt') as in_file:
            for line in in_file:
                output_file_handle.write(line)
                read_count += 0.25  # Each read has 4 lines
    except Exception as e:
        logger.error(f"Error copying FASTQ contents: {e}")
        raise
    
    logger.info(f"Copied {int(read_count)} reads from original dataset")
    
    return int(read_count)