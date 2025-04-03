"""
Coverage bias modeling for the SIRV Integration Pipeline.

This module provides functionality to learn and simulate coverage bias
patterns observed in long-read sequencing, particularly the 5' to 3'
positional bias of reads along transcripts.
"""

import os
import logging
import pickle
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from Bio import SeqIO
import matplotlib.pyplot as plt

# Try to import pysam for BAM processing
try:
    import pysam
except ImportError:
    pysam = None

# Set up logger
logger = logging.getLogger(__name__)


class CoverageBiasModel:
    """
    Model for learning and simulating 5'-3' coverage bias in long reads.
    
    This class learns the positional bias of read starts and ends along
    transcripts from existing data, and can apply similar patterns to
    synthetic reads to better mimic real sequencing characteristics.
    """
    
    def __init__(self, bins: int = 100, smoothing: float = 0.1, 
                 min_coverage: int = 5, seed: Optional[int] = None):
        """
        Initialize the coverage bias model.
        
        Args:
            bins: Number of bins to divide transcripts into for modeling
            smoothing: Strength of smoothing applied to distributions
            min_coverage: Minimum number of reads per transcript to include in model
            seed: Random seed for reproducibility
        """
        self.bins = bins
        self.smoothing = smoothing
        self.min_coverage = min_coverage
        self.has_model = False
        self.start_dist = None
        self.end_dist = None
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def learn_from_bam(self, bam_file: str, reference_file: str) -> bool:
        """
        Learn coverage bias patterns from aligned reads in a BAM file.
        
        Args:
            bam_file: Path to BAM file with reads aligned to transcriptome
            reference_file: Path to reference FASTA file
            
        Returns:
            bool: True if successful, False if failed
        """
        logger.info(f"Learning coverage bias model from BAM file: {bam_file}")
        
        # Check if pysam is available
        if pysam is None:
            logger.error("pysam library is required for BAM processing")
            return False
        
        # Check if files exist
        if not os.path.exists(bam_file):
            logger.error(f"BAM file not found: {bam_file}")
            return False
        
        if not os.path.exists(reference_file):
            logger.error(f"Reference file not found: {reference_file}")
            return False
        
        try:
            # Load reference sequences
            reference_seqs = {}
            for record in SeqIO.parse(reference_file, "fasta"):
                reference_seqs[record.id] = len(record.seq)
            
            if not reference_seqs:
                logger.error(f"No sequences found in reference file: {reference_file}")
                return False
            
            logger.info(f"Loaded {len(reference_seqs)} reference sequences")
            
            # Initialize arrays for start and end positions
            start_positions = []
            end_positions = []
            
            # Process aligned reads
            with pysam.AlignmentFile(bam_file, "rb") as bam:
                # Count reads per transcript
                transcript_reads = {}
                
                for read in bam.fetch():
                    if (read.is_unmapped or read.is_secondary or 
                        read.is_supplementary or not read.reference_name):
                        continue
                    
                    transcript = read.reference_name
                    transcript_reads[transcript] = transcript_reads.get(transcript, 0) + 1
                
                # Process transcripts with sufficient coverage
                for transcript, count in transcript_reads.items():
                    if count < self.min_coverage or transcript not in reference_seqs:
                        continue
                    
                    transcript_length = reference_seqs[transcript]
                    if transcript_length == 0:
                        continue
                    
                    # Process reads from this transcript
                    for read in bam.fetch(transcript):
                        if (read.is_unmapped or read.is_secondary or 
                            read.is_supplementary):
                            continue
                        
                        # Get normalized positions (0-1 range)
                        start_norm = read.reference_start / transcript_length
                        end_norm = read.reference_end / transcript_length
                        
                        start_positions.append(start_norm)
                        end_positions.append(end_norm)
            
            if not start_positions or not end_positions:
                logger.warning("No valid reads found for coverage modeling")
                return self.learn_from_fastq("dummy.fastq")  # Fall back to synthetic model
            
            logger.info(f"Processed {len(start_positions)} reads for coverage modeling")
            
            # Create positional distributions
            self._create_distributions(start_positions, end_positions)
            
            return True
            
        except Exception as e:
            logger.error(f"Error learning coverage bias from BAM: {e}")
            return False
    
    def learn_from_fastq(self, fastq_file: str) -> bool:
        """
        Create a synthetic coverage bias model when BAM alignment is not available.
        
        This creates a model with typical 5'-3' bias patterns observed in
        long-read sequencing without requiring actual alignment data.
        
        Args:
            fastq_file: Path to FASTQ file (only used for logging, not processed)
            
        Returns:
            bool: True if successful
        """
        logger.info(f"Creating synthetic coverage bias model (no alignment available)")
        
        # Create synthetic positions with realistic biases
        # - Start positions tend to favor 5' end 
        # - End positions tend to favor 3' end
        
        # Parameters for beta distributions
        start_alpha, start_beta = 1.5, 3.0  # Skewed toward 5' (smaller values)
        end_alpha, end_beta = 4.0, 1.2      # Skewed toward 3' (larger values)
        
        # Generate synthetic positions
        n_samples = 10000
        start_positions = np.random.beta(start_alpha, start_beta, n_samples)
        end_positions = np.random.beta(end_alpha, end_beta, n_samples)
        
        # Create distributions
        self._create_distributions(start_positions, end_positions)
        
        logger.info("Created synthetic coverage bias model")
        return True
    
    def _create_distributions(self, start_positions: List[float], 
                             end_positions: List[float]) -> None:
        """
        Create normalized, smoothed distributions from position data.
        
        Args:
            start_positions: List of normalized start positions (0-1)
            end_positions: List of normalized end positions (0-1)
        """
        # Create histograms
        start_hist, _ = np.histogram(start_positions, bins=self.bins, range=(0, 1), density=True)
        end_hist, _ = np.histogram(end_positions, bins=self.bins, range=(0, 1), density=True)
        
        # Apply smoothing (add pseudocounts and smooth with running average)
        start_smoothed = self._smooth_distribution(start_hist)
        end_smoothed = self._smooth_distribution(end_hist)
        
        # Normalize to probability distributions
        self.start_dist = start_smoothed / np.sum(start_smoothed)
        self.end_dist = end_smoothed / np.sum(end_smoothed)
        
        self.has_model = True
        
        logger.info("Created positional distributions for coverage bias model")
    
    def _smooth_distribution(self, hist: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to a distribution.
        
        Args:
            hist: Input histogram to smooth
            
        Returns:
            np.ndarray: Smoothed histogram
        """
        # Add pseudocounts to avoid zeros
        smoothed = hist + (np.max(hist) * 0.01)
        
        # Apply moving average smoothing
        window_size = max(3, int(self.bins * self.smoothing))
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(smoothed, window, mode='same')
        
        return smoothed
    
    def plot_distributions(self, output_file: Optional[str] = None) -> bool:
        """
        Plot the coverage bias distributions.
        
        Args:
            output_file: Path to save the plot (optional)
            
        Returns:
            bool: True if successful, False if no model exists
        """
        if not self.has_model:
            logger.warning("Cannot plot distributions: no model exists")
            return False
        
        try:
            # Create bin centers for x-axis
            bin_edges = np.linspace(0, 1, self.bins + 1)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            # Plot start distribution
            plt.plot(bin_centers, self.start_dist, 'b-', linewidth=2, 
                    label="Read start positions (5' bias)")
            
            # Plot end distribution
            plt.plot(bin_centers, self.end_dist, 'r-', linewidth=2,
                    label="Read end positions (3' bias)")
            
            # Add labels and legend
            plt.xlabel('Normalized position along transcript (5\' to 3\')')
            plt.ylabel('Probability density')
            plt.title('Coverage Bias Model: 5\'-3\' Positional Bias')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save plot if output file provided
            if output_file:
                plt.savefig(output_file, dpi=300)
                logger.info(f"Coverage bias plot saved to {output_file}")
            
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Error plotting distributions: {e}")
            return False
    
    def sample_read_position(self, transcript_length: int) -> Tuple[int, int]:
        """
        Sample read start and end positions based on the model.
        
        Args:
            transcript_length: Length of the transcript
            
        Returns:
            Tuple[int, int]: Start and end positions (0-based)
        """
        if not self.has_model:
            # Fallback if no model exists
            start_frac = random.uniform(0, 0.5)  # Slight 5' bias
            end_frac = random.uniform(0.5, 1.0)  # Slight 3' bias
        else:
            # Sample from distributions
            start_bin = np.random.choice(self.bins, p=self.start_dist)
            end_bin = np.random.choice(self.bins, p=self.end_dist)
            
            # Convert bin to fraction, with some noise within the bin
            bin_width = 1.0 / self.bins
            start_frac = (start_bin * bin_width) + (random.random() * bin_width)
            end_frac = (end_bin * bin_width) + (random.random() * bin_width)
        
        # Ensure end is after start
        if end_frac <= start_frac:
            end_frac = start_frac + (random.random() * (1.0 - start_frac))
        
        # Convert to positions
        start_pos = int(start_frac * transcript_length)
        end_pos = int(end_frac * transcript_length)
        
        # Ensure valid positions
        start_pos = max(0, min(start_pos, transcript_length - 1))
        end_pos = max(start_pos + 1, min(end_pos, transcript_length))
        
        return start_pos, end_pos
    
    def apply_to_sequence(self, sequence: str, quality: str, 
                         target_length: Optional[int] = None) -> Tuple[str, str]:
        """
        Apply coverage bias model to extract a realistic fragment from a sequence.
        
        Args:
            sequence: Input sequence
            quality: Quality string for the sequence
            target_length: Target length for the output (optional)
            
        Returns:
            Tuple[str, str]: Extracted sequence and quality fragments
        """
        # For very short sequences, return as-is
        if len(sequence) < 50:
            return sequence, quality
        
        # Get position based on length
        seq_len = len(sequence)
        start, end = self.sample_read_position(seq_len)
        
        # Apply target length if specified
        if target_length is not None and target_length > 0:
            current_len = end - start
            
            if current_len > target_length:
                # Need to shorten
                excess = current_len - target_length
                start_adjust = int(excess * 0.5)
                end_adjust = excess - start_adjust
                
                start += start_adjust
                end -= end_adjust
            
            # Ensure valid range
            start = max(0, start)
            end = min(seq_len, end)
        
        # Extract fragment
        fragment_seq = sequence[start:end]
        fragment_qual = quality[start:end]
        
        return fragment_seq, fragment_qual
    
    def save(self, filename: str) -> bool:
        """
        Save the model to a file.
        
        Args:
            filename: Path to save the model
            
        Returns:
            bool: True if successful, False if no model exists
        """
        if not self.has_model:
            logger.warning("Cannot save: no model exists")
            return False
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'bins': self.bins,
                    'smoothing': self.smoothing,
                    'min_coverage': self.min_coverage,
                    'start_dist': self.start_dist,
                    'end_dist': self.end_dist,
                    'has_model': self.has_model
                }, f)
            
            logger.info(f"Coverage bias model saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @classmethod
    def load(cls, filename: str) -> 'CoverageBiasModel':
        """
        Load a model from a file.
        
        Args:
            filename: Path to the model file
            
        Returns:
            CoverageBiasModel: Loaded model
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file is not a valid model
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
        
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            model = cls(
                bins=data['bins'],
                smoothing=data['smoothing'],
                min_coverage=data['min_coverage']
            )
            
            model.start_dist = data['start_dist']
            model.end_dist = data['end_dist']
            model.has_model = data['has_model']
            
            logger.info(f"Coverage bias model loaded from {filename}")
            return model
            
        except Exception as e:
            raise ValueError(f"Invalid model file: {e}")