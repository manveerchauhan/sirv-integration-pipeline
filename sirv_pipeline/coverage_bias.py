"""
Coverage bias modeling module for SIRV Integration Pipeline.

This module analyzes BAM files to model 5'-3' coverage bias for
transcript integration.
"""

import os
import logging
import subprocess
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

from sirv_pipeline.utils import validate_files

# Set up logger
logger = logging.getLogger(__name__)

# Define full path to samtools
SAMTOOLS_PATH = "/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/SAMtools/1.21/bin/samtools"


def model_transcript_coverage(
    bam_file: str,
    gtf_file: str,
    output_csv: str,
    num_bins: int = 100
) -> pd.DataFrame:
    """Model 5'-3' coverage bias for SIRV transcripts."""
    # Validate input files
    validate_files(bam_file, gtf_file, mode='r')
    validate_files(output_csv, mode='w')
    
    logger.info(f"Modeling transcript coverage from {bam_file}")
    
    # Check if BAM file is empty
    try:
        # Run samtools view to check if BAM has alignments
        result = subprocess.run(
            [SAMTOOLS_PATH, "view", bam_file], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        
        if not result.stdout.strip():
            logger.warning("BAM file has no alignments. Creating default coverage model.")
            # Create a default coverage model with uniform distribution
            transcripts = extract_transcript_coordinates(gtf_file)
            coverage_df = pd.DataFrame(
                {transcript_id: [1.0] * num_bins for transcript_id in transcripts.keys()},
                index=[f"bin_{i+1}" for i in range(num_bins)]
            ).T
            coverage_df.index.name = 'transcript_id'
            coverage_df.to_csv(output_csv)
            logger.info(f"Default coverage model saved to {output_csv}")
            return coverage_df
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking BAM file: {e.stderr}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    
    # Extract transcript coordinates from GTF
    transcripts = extract_transcript_coordinates(gtf_file)
    
    # Extract coverage for each transcript
    coverage_models = {}
    for transcript_id, coords in transcripts.items():
        model = calculate_transcript_coverage(
            bam_file, 
            coords['chrom'], 
            coords['start'], 
            coords['end'], 
            num_bins
        )
        coverage_models[transcript_id] = model
    
    # Save coverage models to CSV
    coverage_df = pd.DataFrame.from_dict(coverage_models, orient='index')
    coverage_df.index.name = 'transcript_id'
    
    # Add column names (bin positions)
    bin_positions = [f"bin_{i+1}" for i in range(num_bins)]
    coverage_df.columns = bin_positions
    
    # Save to CSV
    coverage_df.to_csv(output_csv)
    logger.info(f"Coverage models saved to {output_csv}")
    
    return coverage_df


def extract_transcript_coordinates(gtf_file: str) -> Dict[str, Dict[str, int]]:
    """Extract transcript coordinates from GTF file."""
    transcripts = {}
    
    with open(gtf_file, 'r') as f:
        for line in f:
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Parse GTF line
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            feature_type = fields[2]
            if feature_type != 'transcript':
                continue
            
            # Extract transcript ID
            attributes = fields[8]
            transcript_id = None
            for attr in attributes.split(';'):
                if 'transcript_id' in attr:
                    transcript_id = attr.strip().split(' ')[1].replace('"', '')
                    break
            
            if transcript_id and 'SIRV' in transcript_id:
                # Store transcript coordinates
                transcripts[transcript_id] = {
                    'chrom': fields[0],
                    'start': int(fields[3]),
                    'end': int(fields[4])
                }
    
    logger.info(f"Extracted coordinates for {len(transcripts)} transcripts from GTF")
    return transcripts


def calculate_transcript_coverage(
    bam_file: str,
    chrom: str,
    start: int,
    end: int,
    num_bins: int = 100
) -> List[float]:
    """Calculate positional coverage for a transcript."""
    # Use samtools to extract coverage
    coverage = get_coverage_from_bam(bam_file, chrom, start, end)
    
    # Bin the coverage
    return bin_coverage(coverage, num_bins)


def get_coverage_from_bam(
    bam_file: str,
    chrom: str,
    start: int,
    end: int
) -> List[int]:
    """Get base-level coverage from BAM file using samtools depth."""
    # Build samtools command
    cmd = [
        SAMTOOLS_PATH, "depth",
        "-r", f"{chrom}:{start}-{end}",
        bam_file
    ]
    
    try:
        # Run samtools
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Parse output
        coverage = [0] * (end - start + 1)
        for line in result.stdout.splitlines():
            fields = line.strip().split('\t')
            pos = int(fields[1])
            depth = int(fields[2])
            if start <= pos <= end:
                coverage[pos - start] = depth
        
        return coverage
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running samtools: {e.stderr}")
        return [0] * (end - start + 1)


def bin_coverage(coverage: List[int], num_bins: int) -> List[float]:
    """Bin coverage values into fixed number of bins."""
    if not coverage:
        return [0] * num_bins
    
    # Convert to numpy array
    coverage_array = np.array(coverage)
    
    # Calculate bin size
    bin_size = len(coverage) / num_bins
    
    # Create bins
    binned_coverage = []
    for i in range(num_bins):
        bin_start = int(i * bin_size)
        bin_end = int((i + 1) * bin_size)
        bin_coverage = coverage_array[bin_start:bin_end]
        
        # Calculate mean coverage for bin
        if len(bin_coverage) > 0:
            binned_coverage.append(float(np.mean(bin_coverage)))
        else:
            binned_coverage.append(0.0)
    
    # Normalize bins
    if max(binned_coverage) > 0:
        binned_coverage = [x / max(binned_coverage) for x in binned_coverage]
    
    return binned_coverage


def sample_from_model(
    coverage_model: List[float], 
    transcript_length: int
) -> List[bool]:
    """Sample coverage based on coverage model."""
    # Scale model to transcript length
    scaled_model = scale_model_to_length(coverage_model, transcript_length)
    
    # Sample coverage
    return [np.random.random() < p for p in scaled_model]


def scale_model_to_length(
    coverage_model: List[float],
    target_length: int
) -> List[float]:
    """Scale coverage model to target length."""
    bin_size = target_length / len(coverage_model)
    scaled_model = []
    
    for bin_prob in coverage_model:
        # Repeat bin probability for each position in bin
        num_positions = int(bin_size)
        scaled_model.extend([bin_prob] * num_positions)
    
    # Adjust for rounding errors
    while len(scaled_model) < target_length:
        scaled_model.append(coverage_model[-1])
    
    # Trim if necessary
    return scaled_model[:target_length]


class CoverageBiasModel:
    """Class for modeling and applying coverage bias."""
    
    def __init__(self, coverage_data: Optional[pd.DataFrame] = None, seed: Optional[int] = None):
        """Initialize the coverage bias model."""
        self.coverage_data = coverage_data
        self.has_model = coverage_data is not None
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def apply_to_sequence(self, sequence: str, quality: str, target_length: int) -> Tuple[str, str]:
        """
        Apply coverage bias model to a sequence.
        
        Args:
            sequence: Input sequence string
            quality: Quality string (same length as sequence)
            target_length: Target length for the output sequence
            
        Returns:
            Tuple[str, str]: Modified sequence and quality strings
        """
        if not self.has_model:
            # No model, just truncate to target length
            return sequence[:target_length], quality[:target_length]
        
        # Scale sequence to target length
        if len(sequence) > target_length:
            # Sample from transcript based on coverage model
            coverage_model = self.coverage_data.mean().tolist()
            keep_mask = sample_from_model(coverage_model, len(sequence))
            
            # Apply mask
            new_seq = ''.join([s for s, keep in zip(sequence, keep_mask) if keep])
            new_qual = ''.join([q for q, keep in zip(quality, keep_mask) if keep])
            
            # Trim or pad to target length
            if len(new_seq) > target_length:
                new_seq = new_seq[:target_length]
                new_qual = new_qual[:target_length]
            elif len(new_seq) < target_length:
                # Pad with Ns and low quality scores
                new_seq += 'N' * (target_length - len(new_seq))
                new_qual += '!' * (target_length - len(new_qual))
                
            return new_seq, new_qual
        else:
            # Sequence is shorter than target, just return as is
            return sequence, quality
    
    def plot_distributions(self, output_file: str) -> None:
        """
        Plot coverage distributions.
        
        Args:
            output_file: Path to output plot file
        """
        if not self.has_model:
            logger.warning("No coverage model to plot")
            return
        
        try:
            # Create mean coverage profile
            mean_profile = self.coverage_data.mean()
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(mean_profile.values)
            plt.title('Mean Coverage Profile')
            plt.xlabel('Transcript position (5\' to 3\')')
            plt.ylabel('Relative coverage')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Coverage plot saved to {output_file}")
            
        except ImportError:
            logger.warning("Matplotlib not installed, skipping plot generation")
            

class ReadLengthSampler:
    """Class for sampling read lengths."""
    
    def __init__(self, lengths: List[int], seed: Optional[int] = None):
        """Initialize the read length sampler."""
        self.lengths = lengths
        
        if seed is not None:
            random.seed(seed)
    
    def sample(self) -> int:
        """Sample a read length."""
        return random.choice(self.lengths)


def create_coverage_bias_model(
    fastq_file: str,
    reference_file: Optional[str] = None,
    sample_size: int = 1000,
    seed: Optional[int] = None
) -> Tuple[ReadLengthSampler, CoverageBiasModel]:
    """
    Create coverage bias and read length models from a FASTQ file.
    
    Args:
        fastq_file: Path to FASTQ file to sample reads from
        reference_file: Path to reference FASTA file (optional)
        sample_size: Number of reads to sample
        seed: Random seed for reproducibility
        
    Returns:
        Tuple[ReadLengthSampler, CoverageBiasModel]: Read length sampler and coverage bias model
    """
    logger.info(f"Creating coverage bias model from {fastq_file}")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Validate input files
    validate_files(fastq_file, mode='r')
    
    # Sample read lengths from FASTQ
    lengths = sample_read_lengths(fastq_file, sample_size)
    length_sampler = ReadLengthSampler(lengths, seed=seed)
    logger.info(f"Sampled {len(lengths)} read lengths (mean: {np.mean(lengths):.1f})")
    
    # Create coverage bias model
    coverage_model = None
    
    # Return models
    return length_sampler, CoverageBiasModel(coverage_model, seed=seed)


def sample_read_lengths(fastq_file: str, sample_size: int = 1000) -> List[int]:
    """
    Sample read lengths from a FASTQ file.
    
    Args:
        fastq_file: Path to FASTQ file
        sample_size: Number of reads to sample
        
    Returns:
        List[int]: List of read lengths
    """
    lengths = []
    
    try:
        from Bio import SeqIO
        
        # Determine if gzipped
        opener = open
        if fastq_file.endswith('.gz'):
            import gzip
            opener = gzip.open
        
        # Sample read lengths
        with opener(fastq_file, 'rt') as f:
            records = SeqIO.parse(f, 'fastq')
            
            # Try to get total count for more efficient sampling
            try:
                # Count lines and divide by 4 for FASTQ
                if not fastq_file.endswith('.gz'):
                    with open(fastq_file, 'r') as count_f:
                        line_count = sum(1 for _ in count_f)
                    total_reads = line_count // 4
                    
                    # If sample size is more than half the total, just read all
                    if sample_size > total_reads // 2:
                        for record in records:
                            lengths.append(len(record.seq))
                        return lengths
            except:
                # Just proceed with normal sampling
                pass
            
            # Sample records
            count = 0
            for record in records:
                if random.random() < sample_size / (count + sample_size):
                    if len(lengths) >= sample_size:
                        # Replace a random element
                        idx = random.randint(0, len(lengths) - 1)
                        lengths[idx] = len(record.seq)
                    else:
                        lengths.append(len(record.seq))
                count += 1
                
                # Stop after processing enough reads
                if count >= sample_size * 100 and len(lengths) >= sample_size:
                    break
    except Exception as e:
        logger.error(f"Error sampling read lengths: {e}")
        
    # Ensure we have at least some lengths
    if not lengths:
        logger.warning(f"Could not sample read lengths, using defaults")
        lengths = [1000] * sample_size
        
    return lengths