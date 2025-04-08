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
from scipy import stats
import gffutils
import pysam

from sirv_pipeline.utils import validate_files

# Set up logger
logger = logging.getLogger(__name__)

# Define full path to samtools
SAMTOOLS_PATH = "/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/SAMtools/1.21/bin/samtools"


class CoverageBiasModel:
    """
    Model to capture and simulate 5'-3' coverage bias in cDNA long reads.
    This model specifically targets 10X Chromium cDNA preparation biases.
    """
    
    def __init__(self, model_type="10x_cdna", 
                 bin_count=100, 
                 smoothing_factor=0.05,
                 seed=None):
        """
        Initialize the coverage bias model.
        
        Args:
            model_type: Type of model ('10x_cdna', 'direct_rna', or 'custom')
            bin_count: Number of bins to divide transcripts into
            smoothing_factor: Smoothing factor for kernel density estimation
            seed: Random seed for reproducibility
        """
        # Initialize model parameters
        self.model_type = model_type
        self.bin_count = bin_count
        self.smoothing_factor = smoothing_factor
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize distributions
        self.position_distribution = None  # Overall position distribution
        self.length_dependent_distributions = {}  # Length-stratified distributions
        self.length_bins = []  # Transcript length bins
        
        # Use default model if specified
        if model_type == "10x_cdna":
            self._init_default_10x_cdna_model()
        elif model_type == "direct_rna":
            self._init_default_direct_rna_model()
    
    def _init_default_10x_cdna_model(self):
        """Initialize with default 10X Chromium cDNA bias model."""
        # Implement a default 3' biased distribution (more reads near 1.0 position)
        # This is based on empirical observations from multiple datasets
        
        # Create a 3' biased distribution (more reads near 1.0 position)
        x = np.linspace(0, 1, 1000)
        
        # Stronger bias toward 3' end (position 1.0)
        # Uses beta distribution to model this bias
        y = stats.beta.pdf(x, 1.5, 4.0)  # Parameters tuned for 3' bias
        
        # Normalize
        y = y / np.sum(y)
        
        # Store as position distribution
        self.position_distribution = (x, y)
        
        # Create length-dependent distributions
        # These model how the bias changes with transcript length
        # Shorter transcripts tend to have more uniform coverage
        
        # Define length bins - using fixed values for default model
        self.length_bins = [0, 500, 1000, 2000, 5000, np.inf]
        
        # Create distributions with varying 3' bias by transcript length
        beta_params = [
            (2.0, 2.0),  # Near uniform for very short transcripts
            (1.8, 2.5),  # Slight 3' bias
            (1.6, 3.0),  # Moderate 3' bias
            (1.5, 3.5),  # Stronger 3' bias
            (1.2, 4.0)   # Strongest 3' bias for long transcripts
        ]
        
        # Create distribution for each length bin
        for i, params in enumerate(beta_params):
            a, b = params
            y = stats.beta.pdf(x, a, b)
            y = y / np.sum(y)
            self.length_dependent_distributions[i] = (x, y)
    
    def _init_default_direct_rna_model(self):
        """Initialize with default direct RNA bias model."""
        # Create a 5' biased distribution (more reads near 0.0 position)
        x = np.linspace(0, 1, 1000)
        
        # Parameters tuned for 5' bias (typical in direct RNA)
        y = stats.beta.pdf(x, 2.5, 1.2)
        
        # Normalize
        y = y / np.sum(y)
        
        # Store as position distribution
        self.position_distribution = (x, y)
        
        # Define length bins
        self.length_bins = [0, 500, 1000, 2000, 5000, np.inf]
        
        # Create distributions with varying 5' bias by transcript length
        beta_params = [
            (2.0, 2.0),    # Near uniform for very short transcripts
            (2.2, 1.8),    # Slight 5' bias
            (2.4, 1.5),    # Moderate 5' bias
            (2.5, 1.2),    # Stronger 5' bias
            (3.0, 1.0)     # Strongest 5' bias for long transcripts
        ]
        
        # Create distribution for each length bin
        for i, params in enumerate(beta_params):
            a, b = params
            y = stats.beta.pdf(x, a, b)
            y = y / np.sum(y)
            self.length_dependent_distributions[i] = (x, y)
    
    def learn_from_bam(self, bam_file, annotation_file, 
                       min_reads=100, length_bins=5,
                       transcript_subset=None):
        """
        Learn coverage bias model from aligned BAM file.
        
        Args:
            bam_file: Path to BAM file with aligned reads
            annotation_file: Path to GTF/GFF annotation file
            min_reads: Minimum number of reads for a transcript to be included
            length_bins: Number of transcript length bins to create
            transcript_subset: Optional list of transcript IDs to focus on
            
        Returns:
            self: For method chaining
        """
        # 1. Parse annotation file to get transcript information
        transcripts = self._parse_annotation(annotation_file)
        
        # 2. Process BAM file to extract read positions
        positions = []
        transcript_reads = {}
        
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for read in bam:
                if read.is_unmapped or read.is_supplementary or read.is_secondary:
                    continue
                    
                transcript_id = read.reference_name
                
                # Skip if not in our subset of interest
                if transcript_subset and transcript_id not in transcript_subset:
                    continue
                    
                # Skip if transcript not in annotation
                if transcript_id not in transcripts:
                    continue
                    
                transcript_length = transcripts[transcript_id]['length']
                
                # Skip very short transcripts
                if transcript_length < 200:
                    continue
                    
                # Calculate normalized positions (0-1 range)
                start_norm = read.reference_start / transcript_length
                end_norm = read.reference_end / transcript_length if read.reference_end else start_norm + read.query_length / transcript_length
                
                # Adjust for strand
                if transcripts[transcript_id]['strand'] == '-':
                    start_norm, end_norm = 1 - end_norm, 1 - start_norm
                    
                # Add to overall position distribution
                positions.append((start_norm, end_norm, transcript_length))
                
                # Track by transcript
                if transcript_id not in transcript_reads:
                    transcript_reads[transcript_id] = []
                transcript_reads[transcript_id].append((start_norm, end_norm))
        
        # 3. Filter transcripts with too few reads
        filtered_transcripts = {t: reads for t, reads in transcript_reads.items() 
                              if len(reads) >= min_reads}
        
        # 4. Create overall position distribution
        self.position_distribution = self._create_position_density(
            [p[0] for p in positions]  # Use start positions
        )
        
        # 5. Create length-dependent distributions
        # Group transcripts by length
        transcript_lengths = np.array([t['length'] for t in transcripts.values()])
        length_percentiles = np.percentile(transcript_lengths, 
                                         np.linspace(0, 100, length_bins+1))
        self.length_bins = length_percentiles
        
        # Create distribution for each length bin
        for i in range(length_bins):
            min_len = length_percentiles[i]
            max_len = length_percentiles[i+1]
            
            # Filter positions in this length bin
            bin_positions = [p[0] for p in positions 
                           if min_len <= p[2] < max_len]
            
            if len(bin_positions) > min_reads:
                self.length_dependent_distributions[i] = self._create_position_density(bin_positions)
        
        return self
    
    def apply_bias(self, sequence, read_length=None, transcript_length=None):
        """
        Apply coverage bias model to determine read start position.
        
        Args:
            sequence: Original sequence string
            read_length: Desired read length (or sampled if None)
            transcript_length: Length of original transcript (uses len(sequence) if None)
            
        Returns:
            tuple: (start_position, end_position)
        """
        # Get sequence properties
        if transcript_length is None:
            transcript_length = len(sequence)
            
        # Determine which distribution to use based on transcript length
        dist_idx = 0
        for i, threshold in enumerate(self.length_bins[1:], 1):
            if transcript_length < threshold:
                break
            dist_idx = i
            
        # Use appropriate distribution
        if dist_idx in self.length_dependent_distributions:
            dist = self.length_dependent_distributions[dist_idx]
        else:
            dist = self.position_distribution
            
        # Sample start position
        start_norm = self._sample_from_distribution(dist)
        
        # Convert normalized position to sequence position
        start_pos = int(start_norm * transcript_length)
        
        # Enforce valid range
        start_pos = max(0, min(start_pos, transcript_length - 1))
        
        # Determine end position
        if read_length is None:
            # Use remainder of sequence
            end_pos = transcript_length
        else:
            end_pos = min(start_pos + read_length, transcript_length)
        
        return start_pos, end_pos
    
    def apply_to_sequence(self, sequence, quality, target_length=None):
        """
        Apply coverage bias model to a sequence and its quality scores.
        
        Args:
            sequence: Original sequence string
            quality: Original quality string
            target_length: Target read length (or None to use model)
            
        Returns:
            tuple: (biased_sequence, biased_quality)
        """
        # Apply bias model to get start and end positions
        start_pos, end_pos = self.apply_bias(
            sequence=sequence, 
            read_length=target_length,
            transcript_length=len(sequence)
        )
        
        # Apply bias by trimming the sequence and quality
        biased_seq = sequence[start_pos:end_pos]
        biased_qual = quality[start_pos:end_pos]
        
        return biased_seq, biased_qual
    
    def save(self, filename):
        """Save the bias model to a file."""
        import json
        import datetime
        
        # Convert numpy arrays to lists for JSON serialization
        data = {
            "model_type": self.model_type,
            "bin_count": self.bin_count,
            "smoothing_factor": self.smoothing_factor,
            "position_distribution": {
                "x": self.position_distribution[0].tolist() if self.position_distribution else None,
                "y": self.position_distribution[1].tolist() if self.position_distribution else None
            },
            "length_bins": [float(bin) if bin != np.inf else "Infinity" for bin in self.length_bins],
            "length_dependent_distributions": {
                str(k): {
                    "x": v[0].tolist(),
                    "y": v[1].tolist()
                } for k, v in self.length_dependent_distributions.items()
            },
            "metadata": {
                "created": datetime.datetime.now().isoformat()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Coverage bias model saved to {filename}")
        
        return self
    
    def load(self, filename):
        """Load the bias model from a file."""
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Restore model parameters
        self.model_type = data["model_type"]
        self.bin_count = data["bin_count"]
        self.smoothing_factor = data["smoothing_factor"]
        
        # Restore position distribution
        if data["position_distribution"]["x"]:
            self.position_distribution = (
                np.array(data["position_distribution"]["x"]),
                np.array(data["position_distribution"]["y"])
            )
        
        # Restore length bins
        self.length_bins = [float(bin) if bin != "Infinity" else np.inf 
                           for bin in data["length_bins"]]
        
        # Restore length-dependent distributions
        self.length_dependent_distributions = {}
        for k, v in data["length_dependent_distributions"].items():
            self.length_dependent_distributions[int(k)] = (
                np.array(v["x"]),
                np.array(v["y"])
            )
        
        logger.info(f"Coverage bias model loaded from {filename}")
        
        return self
    
    def plot_distributions(self, output_file=None, show=False):
        """Plot the learned position distributions."""
        plt.figure(figsize=(12, 8))
        
        # Plot overall distribution
        plt.subplot(2, 1, 1)
        if self.position_distribution:
            x, y = self.position_distribution
            plt.plot(x, y, 'k-', linewidth=2, label='Overall')
            plt.title('Overall Coverage Bias Distribution')
            plt.xlabel('Relative Position (5\' → 3\')')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
        
        # Plot length-dependent distributions
        plt.subplot(2, 1, 2)
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.length_dependent_distributions)))
        
        for i, (bin_idx, dist) in enumerate(sorted(self.length_dependent_distributions.items())):
            x, y = dist
            if bin_idx < len(self.length_bins) - 1:
                label = f'{int(self.length_bins[bin_idx])}-{int(self.length_bins[bin_idx+1])} nt'
            else:
                label = f'>{int(self.length_bins[bin_idx])} nt'
            
            plt.plot(x, y, color=colors[i], linewidth=2, label=label)
        
        plt.title('Length-Dependent Coverage Bias Distributions')
        plt.xlabel('Relative Position (5\' → 3\')')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300)
            logger.info(f"Coverage bias plot saved to {output_file}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def _create_position_density(self, positions):
        """Create a kernel density estimate from positions."""
        # Create kernel density estimate
        positions = np.array(positions)
        kde = stats.gaussian_kde(positions, bw_method=self.smoothing_factor)
        
        # Sample points
        x = np.linspace(0, 1, 1000)
        y = kde(x)
        
        # Normalize
        y = y / np.sum(y)
        
        return (x, y)
    
    def _sample_from_distribution(self, distribution):
        """Sample a value from a probability distribution."""
        x, y = distribution
        return np.random.choice(x, p=y/np.sum(y))
    
    def _parse_annotation(self, annotation_file):
        """Parse GTF/GFF annotation file to extract transcript information."""
        # Create in-memory database
        db = gffutils.create_db(annotation_file, ':memory:', merge_strategy='merge')
        
        transcripts = {}
        for transcript in db.features_of_type('transcript'):
            transcript_id = transcript.id.split('.')[0]
            
            # Calculate transcript length
            exons = list(db.children(transcript, featuretype='exon'))
            length = sum(e.end - e.start + 1 for e in exons)
            
            transcripts[transcript_id] = {
                'length': length,
                'strand': transcript.strand
            }
            
        return transcripts


class ReadLengthSampler:
    """Sample read lengths based on observed distributions."""
    
    def __init__(self, lengths=None, min_length=200, max_length=15000, seed=None):
        """
        Initialize the read length sampler.
        
        Args:
            lengths: List of observed read lengths
            min_length: Minimum read length to allow
            max_length: Maximum read length to allow
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.min_length = min_length
        self.max_length = max_length
        
        if lengths and len(lengths) > 10:
            # Filter lengths to min/max range
            filtered_lengths = [l for l in lengths if min_length <= l <= max_length]
            
            if len(filtered_lengths) > 10:
                # Create KDE from observed lengths
                self.kde = stats.gaussian_kde(filtered_lengths, bw_method='scott')
                self.has_kde = True
            else:
                self.has_kde = False
        else:
            self.has_kde = False
    
    def sample(self):
        """Sample a read length from the distribution."""
        if self.has_kde:
            # Sample from KDE
            while True:
                length = int(self.kde.resample(1)[0])
                if self.min_length <= length <= self.max_length:
                    return length
        else:
            # Use simple uniform sampling if no KDE
            return np.random.randint(self.min_length, self.max_length)


def sample_read_lengths(fastq_file, sample_size=1000):
    """Sample read lengths from a FASTQ file."""
    from Bio import SeqIO
    import gzip
    
    lengths = []
    
    # Check if file is gzipped
    opener = gzip.open if fastq_file.endswith('.gz') else open
    
    with opener(fastq_file, 'rt') as f:
        # Sample a subset of reads
        i = 0
        for record in SeqIO.parse(f, 'fastq'):
            lengths.append(len(record.seq))
            i += 1
            if i >= sample_size:
                break
    
    return lengths


def create_coverage_bias_model(
    fastq_file=None,
    bam_file=None,
    annotation_file=None,
    model_type="10x_cdna",
    sample_size=1000,
    min_reads=100,
    length_bins=5,
    seed=None
):
    """
    Create a coverage bias model based on input data or default parameters.
    
    Args:
        fastq_file: FASTQ file to sample read lengths from
        bam_file: BAM file to learn coverage bias from
        annotation_file: GTF/GFF annotation for BAM processing
        model_type: Type of model ('10x_cdna', 'direct_rna', or 'custom')
        sample_size: Number of reads to sample for modeling
        min_reads: Minimum reads per transcript for learning
        length_bins: Number of transcript length bins
        seed: Random seed for reproducibility
        
    Returns:
        Tuple[ReadLengthSampler, CoverageBiasModel]: Length sampler and bias model
    """
    # Initialize coverage bias model
    coverage_model = CoverageBiasModel(model_type=model_type, seed=seed)
    
    # Sample read lengths if FASTQ provided
    if fastq_file:
        logger.info(f"Sampling read lengths from {fastq_file}")
        lengths = sample_read_lengths(fastq_file, sample_size)
        length_sampler = ReadLengthSampler(lengths, seed=seed)
    else:
        length_sampler = ReadLengthSampler(seed=seed)
    
    # Learn coverage bias from BAM if provided
    if bam_file and annotation_file:
        logger.info(f"Learning coverage bias model from {bam_file}")
        try:
            coverage_model.learn_from_bam(
                bam_file=bam_file,
                annotation_file=annotation_file,
                min_reads=min_reads,
                length_bins=length_bins
            )
        except Exception as e:
            logger.error(f"Failed to learn coverage bias model: {e}")
            logger.info("Using default coverage bias model instead")
    
    return length_sampler, coverage_model


# Legacy functions for backward compatibility
def model_transcript_coverage(
    bam_file: str,
    gtf_file: str,
    output_csv: str,
    num_bins: int = 100
) -> pd.DataFrame:
    """Model 5'-3' coverage bias for SIRV transcripts (legacy function)."""
    # Validate input files
    validate_files(bam_file, gtf_file, mode='r')
    validate_files(output_csv, mode='w')
    
    logger.info(f"Modeling transcript coverage from {bam_file}")
    
    # Initialize the new CoverageBiasModel
    coverage_model = CoverageBiasModel(bin_count=num_bins)
    
    try:
        # Learn from BAM
        coverage_model.learn_from_bam(
            bam_file=bam_file,
            annotation_file=gtf_file,
            min_reads=5
        )
        
        # Convert the model to the legacy format
        coverage_df = _convert_model_to_dataframe(coverage_model, num_bins)
        
        # Save to CSV
        coverage_df.to_csv(output_csv)
        logger.info(f"Coverage models saved to {output_csv}")
        
        return coverage_df
        
    except Exception as e:
        logger.error(f"Error modeling transcript coverage: {e}")
        
        # Create a default coverage model with uniform distribution
        transcripts = coverage_model._parse_annotation(gtf_file)
        coverage_df = pd.DataFrame(
            {transcript_id: [1.0] * num_bins for transcript_id in transcripts.keys()},
            index=[f"bin_{i+1}" for i in range(num_bins)]
        ).T
        coverage_df.index.name = 'transcript_id'
        coverage_df.to_csv(output_csv)
        logger.info(f"Default coverage model saved to {output_csv}")
        
        return coverage_df


def _convert_model_to_dataframe(model, num_bins):
    """Convert a CoverageBiasModel to a DataFrame for legacy compatibility."""
    # Get the transcripts from the model
    transcripts = {}
    
    # Use the overall distribution for all transcripts
    if model.position_distribution:
        x, y = model.position_distribution
        
        # Resample to desired number of bins
        bins = np.linspace(0, 1, num_bins)
        
        # Interpolate values at bin positions
        y_interp = np.interp(bins, x, y)
        
        # Normalize
        if np.sum(y_interp) > 0:
            y_interp = y_interp / np.sum(y_interp)
        
        # Create a dataframe with one row per transcript
        # For now, we'll just use the same distribution for all
        coverage_df = pd.DataFrame(
            {i: y_interp for i in range(1)},
            index=[f"bin_{i+1}" for i in range(num_bins)]
        ).T
        
        coverage_df.index.name = 'transcript_id'
        coverage_df.index = ['model']
        
        return coverage_df
    else:
        # Return empty dataframe
        return pd.DataFrame()


def extract_transcript_coordinates(gtf_file: str) -> Dict[str, Dict[str, int]]:
    """Extract transcript coordinates from GTF file (legacy function)."""
    # Use the new CoverageBiasModel's _parse_annotation method
    model = CoverageBiasModel()
    transcripts = model._parse_annotation(gtf_file)
    
    # Convert to legacy format
    legacy_transcripts = {}
    for transcript_id, info in transcripts.items():
        legacy_transcripts[transcript_id] = {
            'chrom': transcript_id.split('.')[0],
            'start': 1,
            'end': info['length']
        }
    
    logger.info(f"Extracted coordinates for {len(legacy_transcripts)} transcripts from GTF")
    return legacy_transcripts


def calculate_transcript_coverage(
    bam_file: str,
    chrom: str,
    start: int,
    end: int,
    num_bins: int = 100
) -> List[float]:
    """Calculate positional coverage for a transcript (legacy function)."""
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
    """Get base-level coverage from BAM file using samtools depth (legacy function)."""
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
    """Bin coverage values into fixed number of bins (legacy function)."""
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
    """Sample coverage based on coverage model (legacy function)."""
    # Scale model to transcript length
    scaled_model = scale_model_to_length(coverage_model, transcript_length)
    
    # Sample coverage
    return [np.random.random() < p for p in scaled_model]


def scale_model_to_length(
    coverage_model: List[float],
    target_length: int
) -> List[float]:
    """Scale coverage model to target length (legacy function)."""
    bin_size = target_length / len(coverage_model)
    scaled_model = []
    
    for bin_prob in coverage_model:
        # Repeat bin probability for each position in bin
        num_positions = int(bin_size)
        scaled_model.extend([bin_prob] * num_positions)
    
    # Adjust for rounding errors
    while len(scaled_model) < target_length:
        scaled_model.append(scaled_model[-1])
    
    # Truncate if too long
    scaled_model = scaled_model[:target_length]
    
    return scaled_model