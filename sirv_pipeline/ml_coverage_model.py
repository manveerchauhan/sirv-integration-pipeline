"""
Advanced machine learning coverage bias model for SIRV Integration Pipeline.

This module provides an ML-based approach to modeling transcript coverage bias,
capable of capturing more complex patterns than the standard parametric model.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Optional, Tuple, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pysam
import re
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

class MLCoverageBiasModel:
    """
    Machine learning model for transcript coverage bias.
    
    This model uses regression techniques to learn complex coverage patterns,
    taking into account transcript features like length, GC content, and
    secondary structure potential.
    """
    
    def __init__(self, 
                 model_type: str = "gradient_boosting",
                 bin_count: int = 100,
                 seed: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the ML coverage bias model.
        
        Args:
            model_type: Type of ML model to use ('gradient_boosting', 'random_forest', or 'ridge')
            bin_count: Number of bins to divide transcripts into
            seed: Random seed for reproducibility
            logger: Logger object for logging
        """
        self.model_type = model_type
        self.bin_count = bin_count
        self.seed = seed
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        # Initialize model
        self.model = self._create_model()
        
        # Initialize transcript features
        self.transcript_features = {}
        
        # Flag to indicate if model is trained
        self.is_trained = False
        
        # Storage for visualization
        self.position_distribution = None
        self.length_dependent_distributions = {}
        self.parameters = {
            "profile": np.ones(1000) / 1000,
            "length_effect": {f"bin_{i+1}": 1.0 for i in range(5)},
            "model_type": "ml_" + model_type
        }
        
    def _create_model(self) -> Pipeline:
        """Create an ML pipeline based on the specified model type."""
        if self.model_type == "gradient_boosting":
            regressor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.seed
            )
        elif self.model_type == "random_forest":
            regressor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.seed
            )
        elif self.model_type == "ridge":
            regressor = Ridge(alpha=1.0, random_state=self.seed)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Create a pipeline with preprocessing steps
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])
        
        return pipeline
    
    def learn_from_bam(self, 
                       bam_file: str, 
                       reference_file: str,
                       annotation_file: str,
                       min_reads: int = 100,
                       feature_extraction: bool = True) -> bool:
        """
        Learn coverage bias patterns from aligned reads.
        
        Args:
            bam_file: Path to BAM file with aligned reads
            reference_file: Path to reference FASTA file
            annotation_file: Path to annotation file (GTF/GFF)
            min_reads: Minimum number of reads per transcript
            feature_extraction: Whether to extract sequence features
            
        Returns:
            bool: True if model was successfully trained
        """
        try:
            self.logger.info(f"Learning ML coverage bias model from BAM file: {bam_file}")
            
            # Extract reference sequences if feature extraction is enabled
            if feature_extraction:
                self._extract_reference_features(reference_file)
                
            # Parse annotation file to get transcript info
            transcript_info = self._parse_annotation(annotation_file)
            if not transcript_info:
                self.logger.error("No transcript information found in annotation file")
                return False
                
            # Extract coverage data from BAM file
            coverage_data = self._extract_coverage_data(bam_file, transcript_info, min_reads)
            if not coverage_data:
                self.logger.error("Failed to extract coverage data from BAM file")
                return False
                
            # Prepare training data
            X, y = self._prepare_training_data(coverage_data, transcript_info)
            if len(X) == 0:
                self.logger.error("No valid training data generated")
                return False
                
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.seed
            )
            
            # Train the model
            self.logger.info(f"Training ML model with {len(X_train)} samples")
            self.model.fit(X_train, y_train)
            
            # Evaluate the model
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            
            self.logger.info(f"Model training complete. R² on training: {train_score:.3f}, validation: {val_score:.3f}")
            
            # Generate visualization data
            self._generate_visualization_data(transcript_info)
            
            # Mark as trained
            self.is_trained = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error learning ML coverage bias model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _extract_reference_features(self, reference_file: str) -> None:
        """
        Extract features from reference sequences.
        
        Args:
            reference_file: Path to reference FASTA file
        """
        try:
            import pyfaidx
            from Bio.SeqUtils import GC
            
            self.logger.info(f"Extracting features from reference sequences: {reference_file}")
            
            # Load FASTA file
            fasta = pyfaidx.Fasta(reference_file)
            
            # Extract features for each sequence
            for seq_name in fasta.keys():
                seq = str(fasta[seq_name])
                length = len(seq)
                
                # Calculate GC content
                gc_content = GC(seq) / 100.0
                
                # Calculate simple complexity measure (proportion of unique kmers)
                k = min(4, length // 2)
                if k > 0:
                    kmers = set()
                    for i in range(length - k + 1):
                        kmers.add(seq[i:i+k])
                    complexity = len(kmers) / (length - k + 1) if length > k else 0
                else:
                    complexity = 0
                
                # Calculate potential for secondary structure (simple heuristic)
                # Count complementary base pairs
                pairs = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
                sec_struct_potential = 0
                for i in range(length - 30):
                    complementary = 0
                    for j in range(10):
                        if i + j < length and i + 20 + j < length:
                            if pairs.get(seq[i + j]) == seq[i + 20 + j]:
                                complementary += 1
                    sec_struct_potential = max(sec_struct_potential, complementary / 10)
                
                # Store features
                self.transcript_features[seq_name] = {
                    'length': length,
                    'gc_content': gc_content,
                    'complexity': complexity,
                    'sec_struct_potential': sec_struct_potential
                }
            
            self.logger.info(f"Extracted features for {len(self.transcript_features)} transcripts")
            
        except Exception as e:
            self.logger.error(f"Error extracting reference features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Continue without features
            self.logger.warning("Continuing without sequence features")
    
    def _parse_annotation(self, annotation_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Parse the annotation file to get transcript information.
        
        Args:
            annotation_file: Path to annotation file (GTF/GFF)
            
        Returns:
            Dictionary mapping transcript IDs to their properties
        """
        result = {}
        
        try:
            with open(annotation_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                        
                    fields = line.strip().split('\t')
                    if len(fields) < 9:
                        continue
                        
                    if fields[2].lower() in ('transcript', 'mrna'):
                        chrom = fields[0]
                        start = int(fields[3])
                        end = int(fields[4])
                        
                        # Try to extract transcript ID from attributes
                        transcript_id = None
                        attributes = fields[8]
                        
                        # Try various formats
                        if 'transcript_id' in attributes:
                            try:
                                transcript_id = attributes.split('transcript_id')[1].split(';')[0].strip().strip('"\'')
                            except:
                                pass
                                
                        # Format with ID=
                        if transcript_id is None and 'ID=' in attributes:
                            try:
                                id_part = [p for p in attributes.split(';') if p.strip().startswith('ID=')]
                                if id_part:
                                    transcript_id = id_part[0].split('=')[1].strip().strip('"\'')
                            except:
                                pass
                                
                        # If we found an ID, store the coordinates
                        if transcript_id:
                            result[transcript_id] = {
                                'chrom': chrom,
                                'start': start,
                                'end': end,
                                'length': end - start + 1
                            }
            
            self.logger.info(f"Parsed {len(result)} transcripts from annotation file")
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing annotation file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def _extract_coverage_data(self, 
                              bam_file: str, 
                              transcript_info: Dict[str, Dict[str, Any]],
                              min_reads: int) -> Dict[str, np.ndarray]:
        """
        Extract coverage data from BAM file.
        
        Args:
            bam_file: Path to BAM file
            transcript_info: Transcript information from annotation
            min_reads: Minimum number of reads per transcript
            
        Returns:
            Dictionary mapping transcript IDs to their coverage arrays
        """
        coverage_data = {}
        
        try:
            # Open BAM file
            bam = pysam.AlignmentFile(bam_file, "rb")
            
            # Process each transcript
            for transcript_id, info in transcript_info.items():
                try:
                    # Get coverage for this transcript
                    coverage = np.zeros(info['length'])
                    
                    # Count reads mapping to this transcript
                    read_count = 0
                    
                    # Try to fetch reads for this transcript
                    try:
                        for read in bam.fetch(transcript_id):
                            read_count += 1
                            # Add coverage for each position
                            for pos in range(read.reference_start, read.reference_end):
                                if 0 <= pos - info['start'] < len(coverage):
                                    coverage[pos - info['start']] += 1
                    except:
                        # Try with chromosome name
                        try:
                            for read in bam.fetch(info['chrom'], info['start'], info['end']):
                                read_count += 1
                                # Add coverage for each position
                                for pos in range(read.reference_start, read.reference_end):
                                    if 0 <= pos - info['start'] < len(coverage):
                                        coverage[pos - info['start']] += 1
                        except:
                            # Skip this transcript
                            continue
                    
                    # Skip transcripts with too few reads
                    if read_count < min_reads:
                        continue
                        
                    # Bin the coverage data
                    binned_coverage = self._bin_coverage(coverage, self.bin_count)
                    
                    # Normalize to mean=1.0
                    if np.mean(binned_coverage) > 0:
                        binned_coverage = binned_coverage / np.mean(binned_coverage)
                        
                    # Store the coverage data
                    coverage_data[transcript_id] = binned_coverage
                    
                except Exception as e:
                    self.logger.debug(f"Error processing transcript {transcript_id}: {str(e)}")
                    continue
            
            bam.close()
            
            self.logger.info(f"Extracted coverage data for {len(coverage_data)} transcripts with at least {min_reads} reads")
            return coverage_data
            
        except Exception as e:
            self.logger.error(f"Error extracting coverage data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def _bin_coverage(self, coverage: np.ndarray, num_bins: int) -> np.ndarray:
        """
        Bin coverage data into a fixed number of bins.
        
        Args:
            coverage: Raw coverage array
            num_bins: Number of bins to use
            
        Returns:
            Binned coverage array
        """
        bin_size = len(coverage) / num_bins
        binned = np.zeros(num_bins)
        
        for i in range(num_bins):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size)
            if end > len(coverage):
                end = len(coverage)
            if start == end:
                binned[i] = 0
            else:
                binned[i] = np.mean(coverage[start:end])
                
        return binned
    
    def _prepare_training_data(self, 
                             coverage_data: Dict[str, np.ndarray],
                             transcript_info: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for the ML model.
        
        Args:
            coverage_data: Dictionary mapping transcript IDs to coverage arrays
            transcript_info: Transcript information from annotation
            
        Returns:
            X, y: Feature matrix and target vector
        """
        # Initialize data storage
        X_data = []
        y_data = []
        
        # Process each transcript
        for transcript_id, coverage in coverage_data.items():
            # Skip if transcript is not in info
            if transcript_id not in transcript_info:
                continue
                
            # Get transcript features
            length = transcript_info[transcript_id]['length']
            
            # Prepare features for each position
            for i in range(len(coverage)):
                # Position as fraction of transcript length
                rel_pos = i / len(coverage)
                
                # Basic features
                features = [rel_pos, length]
                
                # Add sequence features if available
                if transcript_id in self.transcript_features:
                    tf = self.transcript_features[transcript_id]
                    features.extend([
                        tf.get('gc_content', 0.5),
                        tf.get('complexity', 0.5),
                        tf.get('sec_struct_potential', 0.5)
                    ])
                
                # Add position-dependent features
                features.extend([
                    np.sin(2 * np.pi * rel_pos),  # Cyclical representation
                    np.cos(2 * np.pi * rel_pos),
                    rel_pos ** 2,                 # Quadratic term
                    np.sqrt(rel_pos),             # Square root term
                    np.log(rel_pos + 0.01)        # Log term
                ])
                
                # Add to data
                X_data.append(features)
                y_data.append(coverage[i])
        
        # Convert to numpy arrays
        X = np.array(X_data)
        y = np.array(y_data)
        
        return X, y
    
    def _generate_visualization_data(self, transcript_info: Dict[str, Dict[str, Any]]) -> None:
        """
        Generate visualization data for the model.
        
        Args:
            transcript_info: Transcript information from annotation
        """
        # Create position array
        rel_positions = np.linspace(0, 1, 1000)
        
        # Create length groups
        length_groups = [500, 1000, 2000, 5000, 10000]
        
        # Initialize distribution arrays
        position_profile = np.zeros(1000)
        length_distributions = {}
        
        # Generate profile for each position
        for i, pos in enumerate(rel_positions):
            # Basic features
            features = [pos, 2000]  # Use medium length for overall profile
            
            # Add default sequence features
            features.extend([0.5, 0.5, 0.5])  # GC, complexity, sec_struct
            
            # Add position-dependent features
            features.extend([
                np.sin(2 * np.pi * pos),
                np.cos(2 * np.pi * pos),
                pos ** 2,
                np.sqrt(pos),
                np.log(pos + 0.01)
            ])
            
            # Predict coverage
            X = np.array([features])
            position_profile[i] = self.model.predict(X)[0]
        
        # Normalize to mean=1.0
        if np.mean(position_profile) > 0:
            position_profile = position_profile / np.mean(position_profile)
            
        # Store for visualization
        self.position_distribution = (rel_positions, position_profile)
        self.parameters["profile"] = position_profile
        
        # Generate profiles for different transcript lengths
        length_effect = {}
        
        for bin_idx, length in enumerate(length_groups):
            length_profile = np.zeros(1000)
            
            for i, pos in enumerate(rel_positions):
                # Basic features
                features = [pos, length]
                
                # Add default sequence features
                features.extend([0.5, 0.5, 0.5])  # GC, complexity, sec_struct
                
                # Add position-dependent features
                features.extend([
                    np.sin(2 * np.pi * pos),
                    np.cos(2 * np.pi * pos),
                    pos ** 2,
                    np.sqrt(pos),
                    np.log(pos + 0.01)
                ])
                
                # Predict coverage
                X = np.array([features])
                length_profile[i] = self.model.predict(X)[0]
            
            # Normalize to mean=1.0
            if np.mean(length_profile) > 0:
                length_profile = length_profile / np.mean(length_profile)
                
            # Store for visualization
            self.length_dependent_distributions[bin_idx] = (rel_positions, length_profile)
            
            # Calculate bias measure
            bin_5prime = np.sum(length_profile[:len(length_profile)//2])
            bin_3prime = np.sum(length_profile[len(length_profile)//2:])
            
            # Use the 3'/5' ratio as the effect measure
            if bin_5prime > 0:
                effect = bin_3prime / bin_5prime
            else:
                effect = 1.0
                
            length_effect[f"bin_{bin_idx+1}"] = np.float64(effect)
            
        # Store for visualization
        self.parameters["length_effect"] = length_effect
    
    def apply_to_sequence(self, sequence: str, quality: str, target_length: Optional[int] = None) -> Tuple[str, str]:
        """
        Apply coverage bias model to a sequence.
        
        Args:
            sequence: Input DNA/RNA sequence
            quality: Quality string
            target_length: Target output length (if None, will use a random length)
            
        Returns:
            Modified sequence and quality
        """
        # Default to input length if target not specified
        if target_length is None:
            target_length = len(sequence)
            
        # Cap target length to original length
        target_length = min(target_length, len(sequence))
        
        if not self.is_trained:
            self.logger.warning("Model not trained, returning original sequence")
            return sequence[:target_length], quality[:target_length]
        
        try:
            # Get sequence length
            length = len(sequence)
            
            # Calculate features
            gc_content = sum(1 for base in sequence if base in 'GCgc') / length
            
            # Get transcript coverage profile
            rel_positions = np.linspace(0, 1, 1000)
            coverage_profile = np.zeros(1000)
            
            for i, pos in enumerate(rel_positions):
                # Basic features
                features = [pos, length]
                
                # Add sequence features
                features.extend([gc_content, 0.5, 0.5])  # Simplified features
                
                # Add position-dependent features
                features.extend([
                    np.sin(2 * np.pi * pos),
                    np.cos(2 * np.pi * pos),
                    pos ** 2,
                    np.sqrt(pos),
                    np.log(pos + 0.01)
                ])
                
                # Predict coverage
                X = np.array([features])
                coverage_profile[i] = max(0.001, self.model.predict(X)[0])  # Ensure positive
            
            # Convert to probability distribution
            coverage_prob = coverage_profile / np.sum(coverage_profile)
            
            # Sample start position based on coverage profile
            start_idx = np.random.choice(1000, p=coverage_prob)
            start_pos = int(start_idx / 1000 * length)
            
            # Calculate end position based on target length
            end_pos = start_pos + target_length
            
            # Handle boundary conditions
            if end_pos > length:
                # Shift start position to ensure we get target_length bases
                start_pos = max(0, length - target_length)
                end_pos = length
            
            # Extract subsequence
            subseq = sequence[start_pos:end_pos]
            subqual = quality[start_pos:end_pos] if quality else 'I' * (end_pos - start_pos)
            
            return subseq, subqual
            
        except Exception as e:
            self.logger.error(f"Error applying ML coverage bias model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return original sequence as fallback
            return sequence[:target_length], quality[:target_length] if quality else 'I' * target_length
    
    def plot_distributions(self, output_file: Optional[str] = None) -> plt.Figure:
        """
        Plot the coverage bias distribution.
        
        Args:
            output_file: Path to save the plot (optional)
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot coverage profile
        if self.position_distribution is not None:
            x, y = self.position_distribution
        else:
            x = np.linspace(0, 1, 1000)
            y = self.parameters.get("profile", np.ones(1000) / 1000)
            
        ax1.plot(x, y, linewidth=2.5)
        ax1.set_xlabel("Relative position along transcript (5' → 3')", fontsize=12)
        ax1.set_ylabel("Coverage bias factor", fontsize=12)
        ax1.set_title("ML-predicted coverage bias across transcript", fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal line at y=1 (no bias)
        ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7, label="No bias (uniform)")
        
        # Add 5' and 3' end labels
        ax1.text(0.02, 0.02, "5' end", transform=ax1.transAxes, fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        ax1.text(0.98, 0.02, "3' end", transform=ax1.transAxes, fontsize=12, 
                 horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Add explanation for bias
        bias_type = "3' biased" if np.argmax(y) > len(y)/2 else "5' biased"
        ax1.text(0.5, 0.9, f"This model is {bias_type}", transform=ax1.transAxes, 
                 fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        ax1.legend(loc='upper center')
        
        # Plot length effect
        length_effect = self.parameters.get("length_effect", {})
        if length_effect:
            groups = list(length_effect.keys())
            values = [length_effect[g] for g in groups]
            
            # Bar plot for length effect
            bar_positions = np.arange(len(groups))
            bars = ax2.bar(bar_positions, values, color='steelblue')
            ax2.set_xticks(bar_positions)
            
            # Create more informative x-tick labels
            readable_labels = []
            for g in groups:
                if g.startswith('bin_'):
                    bin_num = int(g.replace('bin_', ''))
                    readable_labels.append(f"Group {bin_num}")
                else:
                    readable_labels.append(g)
            
            ax2.set_xticklabels(readable_labels, rotation=45)
            ax2.set_xlabel("Transcript length group", fontsize=12)
            ax2.set_ylabel("Length effect strength", fontsize=12)
            ax2.set_title("Effect of transcript length in ML model", fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add horizontal line at y=1 (no effect)
            ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7, label="No effect")
            ax2.legend()
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Add informative title and explanation
        model_alg = self.model_type.replace('_', ' ').title()
        fig.suptitle(f"ML Coverage Bias Model ({model_alg})", fontsize=16)
        
        # Add a badge to indicate it's an ML model
        fig.text(0.01, 0.97, "ML MODEL", 
                fontsize=10, ha='left', va='top', weight='bold',
                bbox=dict(facecolor='blue', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Add training status badge
        status = "TRAINED" if self.is_trained else "UNTRAINED"
        status_color = "green" if self.is_trained else "red"
        fig.text(0.12, 0.97, status, 
                fontsize=10, ha='left', va='top', weight='bold',
                bbox=dict(facecolor=status_color, alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Add an explanatory text box at the bottom
        explanation = (
            "EXPLANATION: This plot shows the ML-predicted coverage bias.\n"
            "LEFT: Values >1 indicate over-representation, <1 indicate under-representation at that position.\n"
            "RIGHT: How transcript length affects the bias pattern in the ML model."
        )
        fig.text(0.5, 0.01, explanation, ha='center', va='bottom', fontsize=11,
                bbox=dict(facecolor='lightyellow', alpha=0.9, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save figure if output file is specified
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved ML coverage bias plot to {output_file}")
        
        return fig
    
    def save(self, filename: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filename: Path to save the model
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'model_type': self.model_type,
                    'bin_count': self.bin_count,
                    'seed': self.seed,
                    'model': self.model,
                    'transcript_features': self.transcript_features,
                    'is_trained': self.is_trained,
                    'position_distribution': self.position_distribution,
                    'length_dependent_distributions': self.length_dependent_distributions,
                    'parameters': self.parameters
                }, f)
                
            self.logger.info(f"Saved ML coverage bias model to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving ML coverage bias model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    @classmethod
    def load(cls, filename: str) -> 'MLCoverageBiasModel':
        """
        Load a model from a file.
        
        Args:
            filename: Path to the model file
            
        Returns:
            MLCoverageBiasModel: Loaded model
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                
            # Create a new model instance
            model = cls(
                model_type=data.get('model_type', 'gradient_boosting'),
                bin_count=data.get('bin_count', 100),
                seed=data.get('seed', None)
            )
            
            # Load model components
            model.model = data.get('model')
            model.transcript_features = data.get('transcript_features', {})
            model.is_trained = data.get('is_trained', False)
            model.position_distribution = data.get('position_distribution')
            model.length_dependent_distributions = data.get('length_dependent_distributions', {})
            model.parameters = data.get('parameters', {
                "profile": np.ones(1000) / 1000,
                "length_effect": {f"bin_{i+1}": 1.0 for i in range(5)},
                "model_type": "ml_" + model.model_type
            })
            
            logger.info(f"Loaded ML coverage bias model from {filename}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading ML coverage bias model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a new model instance
            return cls() 