"""
Coverage Bias Model for Long-Read RNA Sequencing.

This module provides a model to simulate the 5'-3' coverage bias 
observed in long-read RNA sequencing data, particularly for 
Oxford Nanopore direct RNA sequencing.
"""

import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import pysam
from Bio import SeqIO
import pickle
from typing import Optional, Tuple

# Set up logger
logger = logging.getLogger(__name__)


class CoverageBiasModel:
    """
    A model to simulate 5'-3' coverage bias in long-read RNA sequencing.
    
    The model learns the distribution of read start and end positions 
    along transcripts, allowing for more realistic read simulation.
    """
    
    def __init__(self, bins: int = 100, smoothing: float = 0.1, 
                 min_coverage: int = 5, seed: Optional[int] = None):
        """
        Initialize the coverage bias model.
        
        Args:
            bins: Number of bins to use for position distribution
            smoothing: Smoothing factor for probability distributions
            min_coverage: Minimum read count to consider a position
            seed: Random seed for reproducibility
        """
        self.bins = bins
        self.smoothing = smoothing
        self.min_coverage = min_coverage
        self.has_model = False
        
        # Distributions for read start and end positions
        self.start_dist = None
        self.end_dist = None
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
    
    def learn_from_bam(self, bam_file: str, reference_file: str) -> bool:
        """
        Learn coverage bias from an alignment file.
        
        Args:
            bam_file: Path to BAM file with alignments
            reference_file: Path to reference transcriptome FASTA
            
        Returns:
            bool: True if model was successfully learned
        """
        logger.info(f"Learning coverage bias from {bam_file}...")
        
        try:
            # Load reference transcripts
            references = {record.id: len(record.seq) 
                          for record in SeqIO.parse(reference_file, 'fasta')}
            
            # Prepare distributions
            start_counts = np.zeros(self.bins)
            end_counts = np.zeros(self.bins)
            total_reads = 0
            
            # Open BAM file
            with pysam.AlignmentFile(bam_file, "rb") as bam:
                for read in bam:
                    # Skip problematic reads
                    if (read.is_unmapped or read.is_secondary or 
                        read.is_supplementary or 
                        read.reference_name not in references):
                        continue
                    
                    tx_length = references[read.reference_name]
                    
                    # Normalize start and end positions
                    start_pos = read.reference_start / tx_length
                    end_pos = read.reference_end / tx_length
                    
                    # Bin positions
                    start_bin = int(start_pos * self.bins)
                    end_bin = int(end_pos * self.bins)
                    
                    start_counts[min(start_bin, self.bins-1)] += 1
                    end_counts[min(end_bin, self.bins-1)] += 1
                    total_reads += 1
            
            # Always create a model, even with few reads
            # Add a small constant to prevent zero probabilities
            self.start_dist = (start_counts + self.smoothing) / (max(total_reads, 1) + 2 * self.smoothing)
            self.end_dist = (end_counts + self.smoothing) / (max(total_reads, 1) + 2 * self.smoothing)
            
            # Normalize to ensure sum is 1
            self.start_dist /= self.start_dist.sum()
            self.end_dist /= self.end_dist.sum()
            
            self.has_model = True
            
            # If no reads were found, create a synthetic distribution
            if total_reads == 0:
                logger.warning("No reads found, using synthetic coverage bias model")
                self.learn_from_fastq("dummy.fastq")
            else:
                logger.info("Successfully learned coverage bias model from BAM")
            
            return True
        
        except Exception as e:
            logger.error(f"Error learning coverage bias: {e}")
            return False
    
    def learn_from_fastq(self, fastq_file: str) -> bool:
        """
        Create a synthetic coverage bias model when no BAM is available.
        
        Args:
            fastq_file: Placeholder FASTQ file (not actually used)
            
        Returns:
            bool: Always returns True for synthetic model
        """
        logger.info("Creating synthetic coverage bias model...")
        
        # Create a synthetic bias model
        # Bias towards 5' end at the start
        self.start_dist = np.linspace(1, 0.1, self.bins)
        
        # Bias towards 3' end at the end
        self.end_dist = np.linspace(0.1, 1, self.bins)
        
        # Normalize
        self.start_dist /= self.start_dist.sum()
        self.end_dist /= self.end_dist.sum()
        
        self.has_model = True
        
        logger.info("Synthetic coverage bias model created")
        return True
    
    def sample_read_position(self, transcript_length: int) -> Tuple[int, int]:
        """
        Sample start and end positions for a read.
        
        Args:
            transcript_length: Length of the transcript
            
        Returns:
            Tuple[int, int]: (start_position, end_position)
        """
        if not self.has_model:
            # Default sampling if no model exists
            max_length = 1000  # Typical max read length
            read_length = min(transcript_length, max_length)
            
            start = np.random.randint(0, max(1, transcript_length - read_length))
            end = min(start + read_length, transcript_length)
            
            return start, end
        
        # Use learned distributions
        start_bin = np.random.choice(self.bins, p=self.start_dist)
        end_bin = np.random.choice(self.bins, p=self.end_dist)
        
        start = int(start_bin / self.bins * transcript_length)
        end = int(end_bin / self.bins * transcript_length)
        
        # Ensure end is after start and within transcript
        start = min(start, transcript_length - 1)
        end = max(end, start + 1)
        end = min(end, transcript_length)
        
        return start, end
    
    def apply_to_sequence(self, sequence: str, quality: str, 
                          target_length: Optional[int] = None) -> Tuple[str, str]:
        """
        Apply coverage bias to a sequence, potentially truncating it.
        
        Args:
            sequence: Input sequence
            quality: Quality string for the sequence
            target_length: Optional desired read length
            
        Returns:
            Tuple[str, str]: Truncated sequence and quality string
        """
        # Short sequences get returned unchanged
        if len(sequence) <= 50:
            return sequence, quality
        
        # Use model or default sampling
        if not self.has_model:
            # Simple truncation
            if target_length:
                read_length = min(target_length, len(sequence))
                return sequence[:read_length], quality[:read_length]
            return sequence, quality
        
        # Sample read position
        start, end = self.sample_read_position(len(sequence))
        
        # Optional target length override
        if target_length:
            end = min(start + target_length, end, len(sequence))
        
        # Extract fragment
        result_seq = sequence[start:end]
        result_qual = quality[start:end]
        
        return result_seq, result_qual
    
    def plot_distributions(self, output_file: str) -> bool:
        """
        Plot the learned start and end position distributions.
        
        Args:
            output_file: Path to save the plot
            
        Returns:
            bool: True if plot was successfully created
        """
        if not self.has_model:
            logger.warning("No model available for plotting")
            return False
        
        try:
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.bar(range(self.bins), self.start_dist)
            plt.title("Read Start Position Distribution")
            plt.xlabel("Normalized Transcript Position")
            plt.ylabel("Probability")
            
            plt.subplot(1, 2, 2)
            plt.bar(range(self.bins), self.end_dist)
            plt.title("Read End Position Distribution")
            plt.xlabel("Normalized Transcript Position")
            plt.ylabel("Probability")
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            logger.info(f"Coverage bias plot saved to {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error plotting distributions: {e}")
            return False
    
    def save(self, model_file: str) -> bool:
        """
        Save the current model to a pickle file.
        
        Args:
            model_file: Path to save the model
            
        Returns:
            bool: True if model was successfully saved
        """
        if not self.has_model:
            logger.warning("No model available to save")
            return False
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'start_dist': self.start_dist,
                    'end_dist': self.end_dist,
                    'bins': self.bins,
                    'smoothing': self.smoothing,
                    'min_coverage': self.min_coverage
                }, f)
            
            logger.info(f"Model saved to {model_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @classmethod
    def load(cls, model_file: str) -> 'CoverageBiasModel':
        """
        Load a previously saved model.
        
        Args:
            model_file: Path to the saved model file
            
        Returns:
            CoverageBiasModel: Loaded model instance
        """
        try:
            with open(model_file, 'rb') as f:
                data = pickle.load(f)
            
            model = cls(
                bins=data['bins'],
                smoothing=data['smoothing'],
                min_coverage=data['min_coverage']
            )
            
            model.start_dist = data['start_dist']
            model.end_dist = data['end_dist']
            model.has_model = True
            
            return model
        
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")