"""
Advanced machine learning coverage bias model for RNA-seq data

This module provides a more sophisticated coverage bias model that uses
random forest regression to learn complex patterns in RNA-seq coverage data.
It extracts features from sequences and transcript properties to predict
coverage at different positions along transcripts.

This is particularly useful for long-read RNA-seq data where coverage biases
can be complex and dependent on multiple factors.
"""

import os
import sys
import logging
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from sklearn.model_selection import learning_curve
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import tempfile
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set up logger
logger = logging.getLogger(__name__)

class MLCoverageBiasModel:
    """
    Machine learning-based coverage bias model using random forest regression or gradient boosting.
    
    This model analyzes BAM files to learn complex coverage patterns across
    transcripts using machine learning. It can:
    1. Extract sequence features from reference FASTA
    2. Learn transcript-specific and position-specific coverage biases
    3. Generate accurate coverage profiles for any transcript
    4. Visualize learned coverage patterns
    """
    
    def __init__(self, 
                 model_type: str = "random_forest",
                 bin_count: int = 100,
                 seed: Optional[int] = None,
                 logger: Optional[logging.Logger] = None,
                 config: Optional[Dict[str, Any]] = None,
                 config_preset: Optional[str] = None,
                 config_file: Optional[str] = None,
                 feature_cache: Optional[Any] = None,
                 feature_cache_file: Optional[str] = None):
        """
        Initialize the ML coverage bias model.
        
        Args:
            model_type: Type of ML model ('random_forest' or 'ml_gradient_boosting')
            bin_count: Number of bins for coverage profile
            seed: Random seed for reproducibility
            logger: Logger object
            config: Configuration dictionary for model hyperparameters
            config_preset: Preset configuration name
            config_file: Path to configuration file
            feature_cache: Pre-initialized feature cache object
            feature_cache_file: Path to feature cache file
        """
        # Model type - support random_forest and gradient_boosting
        self.model_type = model_type if model_type in ["random_forest", "ml_gradient_boosting"] else "random_forest"
        
        # Basic parameters
        self.bin_count = bin_count
        self.seed = seed
        self.is_trained = False
        
        # Set logger
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize feature dict
        self.transcript_features = {}
        
        # Initialize distributions
        self.position_distribution = None
        self.length_dependent_distributions = {}
        
        # Initialize parameters for serialization
        self.parameters = {
            "profile": np.ones(1000) / 1000,
            "length_effect": {f"bin_{i+1}": 1.0 for i in range(5)},
            "model_type": self.model_type
        }
        
        # Training data storage for visualization
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.feature_names = []
        
        # Set up feature cache if needed
        if feature_cache is not None:
            self.feature_cache = feature_cache
        elif feature_cache_file is not None:
            try:
                from sirv_pipeline.feature_cache import FeatureCache
                self.feature_cache = FeatureCache(feature_cache_file)
                self.logger.info(f"Initialized feature cache from {feature_cache_file}")
            except ImportError:
                self.logger.warning("Feature cache module not available, proceeding without cache")
                self.feature_cache = None
        else:
            self.feature_cache = None
            
        # Load configuration
        if config is not None:
            self.config = config
        else:
            try:
                from sirv_pipeline.model_config import get_model_config
                self.config = get_model_config(
                    preset_name=config_preset, 
                    config_file=config_file
                )
                self.logger.info(f"Loaded model configuration with {len(self.config)} parameters")
            except ImportError:
                self.logger.warning("Model configuration module not available, using default settings")
                if self.model_type == "random_forest":
                    self.config = {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                        "bootstrap": True
                    }
                else:  # gradient_boosting
                    self.config = {
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "max_depth": 3,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                        "max_features": "sqrt",
                        "subsample": 0.8
                    }
                
        # Initialize model
        self.model = self._create_model()
        
    def _create_model(self) -> Pipeline:
        """
        Create the machine learning model pipeline.
        
        Returns:
            Pipeline: Scikit-learn pipeline with feature transformation and model
        """
        if self.model_type == "random_forest":
            # Extract random forest parameters
            n_estimators = self.config.get("n_estimators", 100)
            max_depth = self.config.get("max_depth", 10)
            min_samples_split = self.config.get("min_samples_split", 2)
            min_samples_leaf = self.config.get("min_samples_leaf", 1)
            max_features = self.config.get("max_features", "sqrt")
            bootstrap = self.config.get("bootstrap", True)
            
            # Create random forest regressor
            regressor = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=self.seed,
                n_jobs=-1,  # Use all CPUs
                verbose=0,
                oob_score=True  # Enable out-of-bag score estimation
            )
            
            self.logger.info(f"Created random forest model with {n_estimators} trees and max depth {max_depth}")
        
        elif self.model_type == "ml_gradient_boosting":
            # Extract gradient boosting parameters
            n_estimators = self.config.get("n_estimators", 100)
            learning_rate = self.config.get("learning_rate", 0.1)
            max_depth = self.config.get("max_depth", 3)
            min_samples_split = self.config.get("min_samples_split", 2)
            min_samples_leaf = self.config.get("min_samples_leaf", 1)
            max_features = self.config.get("max_features", "sqrt")
            subsample = self.config.get("subsample", 0.8)
            
            # Create gradient boosting regressor
            regressor = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                subsample=subsample,
                random_state=self.seed,
                verbose=0
            )
            
            self.logger.info(f"Created gradient boosting model with {n_estimators} trees, learning rate {learning_rate}, and max depth {max_depth}")
        
        # Create pipeline with polynomial features
        # This allows learning complex interactions between features
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=True)),
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
        Learn coverage bias model from a BAM file.
        
        Args:
            bam_file: Path to BAM file with RNA-seq reads
            reference_file: Path to reference FASTA file
            annotation_file: Path to annotation file (GTF/GFF)
            min_reads: Minimum reads per transcript for analysis
            feature_extraction: Whether to extract sequence features
            
        Returns:
            bool: Success status
        """
        self.logger.info(f"Learning coverage bias from BAM file: {bam_file}")
        
        try:
            # Extract features from reference if available
            if feature_extraction and reference_file:
                self._extract_reference_features(reference_file)
            else:
                self.logger.info("Skipping feature extraction")
            
            # Parse annotation to get transcript information
            transcript_info = self._parse_annotation(annotation_file)
            
            if not transcript_info:
                self.logger.error("Failed to parse annotation file")
                return False
            
            self.logger.info(f"Extracted information for {len(transcript_info)} transcripts")
            
            # Extract coverage data from BAM file
            coverage_data = self._extract_coverage_data(bam_file, transcript_info, min_reads)
            
            if not coverage_data:
                self.logger.error("Failed to extract coverage data from BAM file")
                return False
            
            self.logger.info(f"Extracted coverage data for {len(coverage_data)} transcripts")
            
            # Prepare training data
            X, y = self._prepare_training_data(coverage_data, transcript_info)
            
            if len(X) == 0 or len(y) == 0:
                self.logger.error("No valid training data generated")
                return False
            
            self.logger.info(f"Prepared {X.shape[0]} training samples with {X.shape[1]} features")
            
            # Generate feature names for visualization
            self._generate_feature_names(X.shape[1])
            
            # Split data into training and validation sets
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.seed
            )
            
            # Train the model
            self.logger.info("Training random forest model...")
            start_time = time.time()
            self.model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            # Evaluate model
            y_pred = self.model.predict(self.X_val)
            mse = mean_squared_error(self.y_val, y_pred)
            r2 = r2_score(self.y_val, y_pred)
            
            self.logger.info(f"Model training completed in {training_time:.2f} seconds")
            self.logger.info(f"Model performance: MSE={mse:.4f}, R²={r2:.4f}")
            
            # Log feature importances if available
            rf_model = self.model.named_steps['regressor']
            importances = rf_model.feature_importances_
            if importances is not None and len(importances) > 0:
                top_indices = np.argsort(importances)[-5:]  # Get indices of top 5 features
                self.logger.info("Top 5 most important features for coverage prediction:")
                for i in reversed(top_indices):
                    if i < len(importances):
                        self.logger.info(f"Feature {i}: Importance {importances[i]:.4f}")
            
            # Generate visualization data
            self._generate_visualization_data(transcript_info)
            
            # Mark as trained
            self.is_trained = True
            
            # Save feature cache if available
            if self.feature_cache is not None:
                self.feature_cache.save_to_file()
                cache_stats = self.feature_cache.get_stats()
                self.logger.info(f"Feature cache stats: {cache_stats['size']} features, "
                                 f"{cache_stats['hits']} hits, {cache_stats['misses']} misses")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error learning coverage model: {str(e)}")
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
            
            self.logger.info(f"Extracting RNA-seq relevant features from reference sequences: {reference_file}")
            
            # Load FASTA file
            fasta = pyfaidx.Fasta(reference_file)
            
            # Extract features for each sequence
            total_sequences = len(fasta.keys())
            processed = 0
            cached = 0
            
            for seq_name in fasta.keys():
                processed += 1
                
                # Check if features are already in cache
                if self.feature_cache is not None and seq_name in self.feature_cache:
                    self.transcript_features[seq_name] = self.feature_cache.get(seq_name)
                    cached += 1
                    
                    # Log progress periodically
                    if processed % 100 == 0 or processed == total_sequences:
                        self.logger.info(f"Processed {processed}/{total_sequences} sequences, {cached} from cache")
                    
                    continue
                
                # Extract sequence
                seq = str(fasta[seq_name])
                length = len(seq)
                
                if length < 10:  # Skip very short sequences
                    continue
                
                # Calculate features
                if self.feature_cache is not None:
                    # Use feature cache to calculate and store features
                    features = self.feature_cache.add_sequence_features(seq_name, seq)
                    self.transcript_features[seq_name] = features
                else:
                    # Calculate features manually if no cache
                    features = self._calculate_sequence_features(seq)
                    self.transcript_features[seq_name] = features
                
                # Log progress periodically
                if processed % 100 == 0 or processed == total_sequences:
                    self.logger.info(f"Processed {processed}/{total_sequences} sequences, {cached} from cache")
            
            self.logger.info(f"Extracted features for {len(self.transcript_features)} transcripts "
                            f"({cached} from cache, {len(self.transcript_features) - cached} newly calculated)")
            
            # Save cache if available
            if self.feature_cache is not None:
                self.feature_cache.save_to_file()
            
        except Exception as e:
            self.logger.error(f"Error extracting reference features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Continue without features
            self.logger.warning("Continuing without sequence features")
    
    def _calculate_sequence_features(self, sequence: str) -> Dict[str, Any]:
        """
        Calculate features for a sequence when feature cache is not available.
        
        Args:
            sequence: Nucleotide sequence
            
        Returns:
            Dictionary of calculated features
        """
        sequence = sequence.upper()  # Normalize sequence
        length = len(sequence)
        
        if length == 0:
            return {
                'length': 0,
                'gc_content': 0.0,
                'five_prime_gc': 0.0,
                'three_prime_gc': 0.0,
                'complexity': 0.0,
                'hairpin_potential': 0.0,
                'g_quadruplex_potential': 0.0,
                'homopolymer_ratio': 0.0,
                'gc_skew': 0.0
            }
        
        # Basic composition
        g_count = sequence.count('G')
        c_count = sequence.count('C')
        gc_count = g_count + c_count
        gc_content = gc_count / length
        
        # GC skew
        gc_skew = (g_count - c_count) / (g_count + c_count) if (g_count + c_count) > 0 else 0
        
        # Terminal regions
        five_prime_region = sequence[:min(50, length)]
        three_prime_region = sequence[max(0, length - 50):]
        
        five_prime_gc = (five_prime_region.count('G') + five_prime_region.count('C')) / len(five_prime_region) if five_prime_region else 0
        three_prime_gc = (three_prime_region.count('G') + three_prime_region.count('C')) / len(three_prime_region) if three_prime_region else 0
        
        # Sequence complexity (k-mer diversity)
        k = min(4, length // 2) if length > 4 else 1
        kmers = set()
        for i in range(length - k + 1):
            kmers.add(sequence[i:i+k])
        complexity = len(kmers) / (length - k + 1) if length > k else 0
        
        # RNA structure potential
        pairs = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        
        # Hairpin potential
        hairpin_potential = 0
        for i in range(length - 30):
            complementary = 0
            for j in range(10):
                if i + j < length and i + 20 + j < length:
                    if pairs.get(sequence[i + j]) == sequence[i + 20 + j]:
                        complementary += 1
            hairpin_potential = max(hairpin_potential, complementary / 10)
        
        # G-quadruplex potential
        g_runs = 0
        for i in range(length - 15):
            window = sequence[i:i+15]
            g_count = window.count('G')
            if g_count > 10:  # High G content in small window
                g_runs += 1
        g_quadruplex_potential = g_runs / (length - 14) if length > 14 else 0
        
        # Homopolymer content
        max_homopolymer = 0
        for base in 'ACGT':
            i = 0
            while i < length:
                run_length = 0
                while i < length and sequence[i] == base:
                    run_length += 1
                    i += 1
                max_homopolymer = max(max_homopolymer, run_length)
                if i < length and sequence[i] != base:
                    i += 1
        homopolymer_ratio = max_homopolymer / length if length > 0 else 0
        
        # Return features
        return {
            'length': length,
            'gc_content': gc_content,
            'gc_skew': gc_skew,
            'complexity': complexity,
            'five_prime_gc': five_prime_gc,
            'three_prime_gc': three_prime_gc,
            'hairpin_potential': hairpin_potential,
            'g_quadruplex_potential': g_quadruplex_potential,
            'homopolymer_ratio': homopolymer_ratio
        }
    
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
        
        # Get feature weights from config
        feature_weights = self.config.get("feature_weights", {})
        
        # Process each transcript
        for transcript_id, coverage in coverage_data.items():
            # Skip if transcript is not in info
            if transcript_id not in transcript_info:
                continue
                
            # Get transcript features
            length = transcript_info[transcript_id]['length']
            
            # Get additional sequence features if available
            seq_features = {}
            if transcript_id in self.transcript_features:
                seq_features = self.transcript_features[transcript_id]
                
            # Log scale transcript length - more appropriate for RNA-seq data
            # RNA-seq coverage patterns often depend on log-length
            log_length = np.log1p(length)
            
            # Prepare features for each position
            for i in range(len(coverage)):
                # Position as fraction of transcript length
                rel_pos = i / len(coverage)
                
                # Basic features - always included
                features = [rel_pos, log_length]
                
                # Add common sequence features if available
                if 'gc_content' in seq_features:
                    gc_weight = feature_weights.get("gc_content", 1.0)
                    features.append(seq_features['gc_content'] * gc_weight)
                    
                    # Interaction term between position and GC content
                    # Coverage bias often depends on position-dependent GC effects
                    features.append(rel_pos * seq_features['gc_content'] * gc_weight)
                else:
                    # Placeholder values if features are missing
                    features.extend([0.5, 0.5 * rel_pos])
                
                # Add advanced RNA-seq specific features if available
                if seq_features:
                    # Terminal region features - critical for modeling 5'/3' bias
                    if 'five_prime_gc' in seq_features:
                        five_prime_weight = feature_weights.get("five_prime_gc", 1.0)
                        features.append(seq_features['five_prime_gc'] * five_prime_weight)
                        # Interaction with position (stronger effect near 5' end)
                        features.append((1.0 - rel_pos) * seq_features['five_prime_gc'] * five_prime_weight)
                    else:
                        features.extend([0.5, 0.5 * (1.0 - rel_pos)])
                        
                    if 'three_prime_gc' in seq_features:
                        three_prime_weight = feature_weights.get("three_prime_gc", 1.0)
                        features.append(seq_features['three_prime_gc'] * three_prime_weight)
                        # Interaction with position (stronger effect near 3' end)
                        features.append(rel_pos * seq_features['three_prime_gc'] * three_prime_weight)
                    else:
                        features.extend([0.5, 0.5 * rel_pos])
                    
                    # Structural features - affect RT efficiency position-dependently
                    if 'hairpin_potential' in seq_features:
                        hairpin_weight = feature_weights.get("hairpin_potential", 1.0)
                        features.append(seq_features['hairpin_potential'] * hairpin_weight)
                        # Hairpins have stronger effects in the middle of transcripts
                        mid_pos_factor = 1.0 - abs(rel_pos - 0.5) * 2  # highest at rel_pos=0.5
                        features.append(mid_pos_factor * seq_features['hairpin_potential'] * hairpin_weight)
                    else:
                        features.extend([0.25, 0.25 * (1.0 - abs(rel_pos - 0.5) * 2)])
                    
                    # G-quadruplex features - especially important for nanopore RNA-seq
                    if 'g_quadruplex_potential' in seq_features:
                        gquad_weight = feature_weights.get("g_quadruplex_potential", 1.0)
                        features.append(seq_features['g_quadruplex_potential'] * gquad_weight)
                    else:
                        features.append(0.1)
                        
                    # Homopolymer and repeat features - crucial for long-read accuracy
                    if 'homopolymer_ratio' in seq_features:
                        homo_weight = feature_weights.get("homopolymer_ratio", 1.0)
                        features.append(seq_features['homopolymer_ratio'] * homo_weight)
                    else:
                        features.append(0.1)
                        
                    # Complexity feature - relates to mappability and coverage
                    if 'complexity' in seq_features:
                        complexity_weight = feature_weights.get("complexity", 1.0)
                        features.append(seq_features['complexity'] * complexity_weight)
                        # Higher complexity regions tend to have more uniform coverage
                        features.append(seq_features['complexity'] * (1.0 - abs(rel_pos - 0.5) * 2) * complexity_weight)
                    else:
                        features.extend([0.5, 0.5 * (1.0 - abs(rel_pos - 0.5) * 2)])
                        
                    # GC skew feature - important for strand-specific protocols
                    if 'gc_skew' in seq_features:
                        gc_skew_weight = feature_weights.get("gc_skew", 1.0)
                        features.append(seq_features['gc_skew'] * gc_skew_weight)
                        # GC skew has position-dependent effects
                        features.append(rel_pos * seq_features['gc_skew'] * gc_skew_weight)
                    else:
                        features.extend([0.0, 0.0])
                
                # Get the coverage value for this position
                cov_value = coverage[i]
                
                # Store this data point
                X_data.append(features)
                y_data.append(cov_value)
        
        # Convert to numpy arrays
        X = np.array(X_data)
        y = np.array(y_data)
        
        self.logger.info(f"Prepared training data: {X.shape[0]} samples with {X.shape[1]} features per sample")
        
        # Handle empty datasets
        if len(X) == 0:
            self.logger.warning("No training data was generated")
            return np.array([]), np.array([])
            
        return X, y
    
    def _generate_visualization_data(self, transcript_info: Dict[str, Dict[str, Any]]) -> None:
        """
        Generate visualization data for the random forest model.
        
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
        
        # Get the feature names if possible
        feature_names = []
        try:
            poly = self.model.named_steps['poly']
            # Create sample feature
            basic_feature = np.array([[0.5, 2000, 0.5, 0.5, 0.5]])  # Position, length, GC, complexity, etc.
            transformed = poly.transform(basic_feature)
            feature_names = [f"Feature_{i}" for i in range(transformed.shape[1])]
        except:
            pass
        
        # Generate profile for each position
        for i, pos in enumerate(rel_positions):
            # Sample features for this position
            # Use log-transformed length for predictions
            log_length = np.log1p(2000)  # Medium transcript length
            
            # Create input features with standard values
            # The exact features depend on the model's training data
            features = [pos, log_length]  # Position and log length
            
            # Add placeholder sequence features
            features.extend([0.5, 0.5 * pos])  # GC content and interaction
            
            # Add placeholder terminal features
            features.extend([0.5, 0.5 * (1.0 - pos)])  # 5' GC and interaction
            features.extend([0.5, 0.5 * pos])  # 3' GC and interaction
            
            # Add placeholder structural features
            features.extend([0.3, 0.3 * (1.0 - abs(pos - 0.5) * 2)])  # Hairpin potential
            features.extend([0.1, 0.2, 0.1, 0.5, 0.5 * (1.0 - abs(pos - 0.5) * 2)])  # Other features
            features.extend([0.0, 0.0])  # GC skew features
            
            # Predict coverage using the model
            try:
                X = np.array([features])
                position_profile[i] = max(0.01, self.model.predict(X)[0])  # Ensure positive values
            except Exception as e:
                self.logger.warning(f"Error predicting coverage for position {pos}: {str(e)}")
                position_profile[i] = 1.0  # Default to uniform if prediction fails
        
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
            log_length = np.log1p(length)
            
            for i, pos in enumerate(rel_positions):
                # Create features for this position and length
                features = [pos, log_length]
                
                # Add placeholder sequence features
                features.extend([0.5, 0.5 * pos])  # GC content and interaction
                
                # Add placeholder terminal features
                features.extend([0.5, 0.5 * (1.0 - pos)])  # 5' GC and interaction
                features.extend([0.5, 0.5 * pos])  # 3' GC and interaction
                
                # Add placeholder structural features
                features.extend([0.3, 0.3 * (1.0 - abs(pos - 0.5) * 2)])  # Hairpin potential
                features.extend([0.1, 0.2, 0.1, 0.5, 0.5 * (1.0 - abs(pos - 0.5) * 2)])  # Other features
                features.extend([0.0, 0.0])  # GC skew features
                
                # Predict coverage using the model
                try:
                    X = np.array([features])
                    length_profile[i] = max(0.01, self.model.predict(X)[0])
                except:
                    length_profile[i] = 1.0  # Default to uniform
            
            # Normalize to mean=1.0
            if np.mean(length_profile) > 0:
                length_profile = length_profile / np.mean(length_profile)
                
            # Store for visualization
            self.length_dependent_distributions[bin_idx] = (rel_positions, length_profile)
            
            # Calculate bias measure - 3'/5' bias ratio
            bin_5prime = np.sum(length_profile[:len(length_profile)//2])
            bin_3prime = np.sum(length_profile[len(length_profile)//2:])
            
            # Calculate effect measure (3'/5' ratio)
            if bin_5prime > 0:
                effect = bin_3prime / bin_5prime
            else:
                effect = 1.0
                
            length_effect[f"bin_{bin_idx+1}"] = np.float64(effect)
            
        # Store for visualization
        self.parameters["length_effect"] = length_effect
        self.parameters["model_type"] = "random_forest"  # Ensure correct model type
    
    def plot_distributions(self, output_file: Optional[str] = None) -> plt.Figure:
        """
        Plot the coverage bias distribution learned by the random forest model.
        
        Args:
            output_file: Path to save the plot (optional)
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot coverage profile
        if self.position_distribution is not None:
            x, y = self.position_distribution
        else:
            x = np.linspace(0, 1, 1000)
            y = self.parameters.get("profile", np.ones(1000) / 1000)
            
        ax1.plot(x, y, linewidth=2.5, color='#1f77b4')
        ax1.set_xlabel("Relative position along transcript (5' → 3')", fontsize=12)
        ax1.set_ylabel("Coverage bias factor", fontsize=12)
        ax1.set_title("Random Forest predicted coverage bias", fontsize=14)
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
            # Use a color gradient to indicate shorter to longer transcripts
            colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
            bars = ax2.bar(bar_positions, values, color=colors)
            ax2.set_xticks(bar_positions)
            
            # Create more informative x-tick labels
            readable_labels = []
            for i, g in enumerate(groups):
                if g.startswith('bin_'):
                    bin_num = int(g.replace('bin_', ''))
                    if bin_num == 1:
                        readable_labels.append("Shortest")
                    elif bin_num == len(groups):
                        readable_labels.append("Longest")
                    else:
                        readable_labels.append(f"Length\nGroup {bin_num}")
                else:
                    readable_labels.append(g)
            
            ax2.set_xticklabels(readable_labels, rotation=0)
            ax2.set_xlabel("Transcript length groups", fontsize=12)
            ax2.set_ylabel("Length effect factor (3'/5' ratio)", fontsize=12)
            ax2.set_title("Effect of transcript length on coverage", fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add horizontal line at y=1 (no effect)
            ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7, label="No length effect")
            ax2.legend()
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=10)
                        
            # Add gradient color legend for transcript length
            gradient_ax = fig.add_axes([0.92, 0.4, 0.01, 0.2])
            gradient = np.linspace(0, 1, 100).reshape(-1, 1)
            gradient_ax.imshow(gradient, aspect='auto', cmap=plt.cm.viridis)
            gradient_ax.set_xticks([])
            gradient_ax.set_yticks([0, 99])
            gradient_ax.set_yticklabels(['Shorter', 'Longer'], fontsize=10)
            gradient_ax.set_title('Transcript\nLength', fontsize=10)
        
        # Add informative title
        fig.suptitle("Random Forest Coverage Bias Model", fontsize=16)
        
        # Add a badge to indicate it's an ML model
        fig.text(0.01, 0.97, "ML MODEL", 
                fontsize=10, ha='left', va='top', weight='bold', color='white',
                bbox=dict(facecolor='blue', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Add training status badge
        status = "TRAINED" if self.is_trained else "UNTRAINED"
        status_color = "green" if self.is_trained else "red"
        fig.text(0.12, 0.97, status, 
                fontsize=10, ha='left', va='top', weight='bold', color='white',
                bbox=dict(facecolor=status_color, alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Add an explanatory text box at the bottom
        explanation = (
            "This model uses random forest machine learning to predict complex RNA-seq coverage patterns.\n"
            "LEFT: Values >1 indicate over-representation at that position, <1 indicate under-representation.\n"
            "RIGHT: Shows how bias changes with transcript length (values >1 indicate stronger 3' bias)."
        )
        fig.text(0.5, 0.01, explanation, ha='center', va='bottom', fontsize=11,
                bbox=dict(facecolor='lightyellow', alpha=0.9, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.05, 0.92, 0.95])
        
        # Save figure if output file is specified
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved random forest coverage bias plot to {output_file}")
        
        return fig
    
    def save(self, filename: str) -> None:
        """
        Save the random forest model to a file.
        
        Args:
            filename: Path to save the model
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'model_type': 'random_forest',
                    'bin_count': self.bin_count,
                    'seed': self.seed,
                    'model': self.model,
                    'transcript_features': self.transcript_features,
                    'is_trained': self.is_trained,
                    'position_distribution': self.position_distribution,
                    'length_dependent_distributions': self.length_dependent_distributions,
                    'parameters': self.parameters,
                    'config': self.config
                }, f)
                
            self.logger.info(f"Saved random forest coverage model to {filename}")
            
            # Save feature cache if available
            if self.feature_cache is not None and hasattr(self.feature_cache, 'cache_file') and self.feature_cache.cache_file:
                self.feature_cache.save_to_file()
                self.logger.info(f"Saved feature cache to {self.feature_cache.cache_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    @classmethod
    def load(cls, filename: str, feature_cache_file: Optional[str] = None) -> 'MLCoverageBiasModel':
        """
        Load a model from a file.
        
        Args:
            filename: Path to the model file
            feature_cache_file: Optional path to feature cache file
            
        Returns:
            MLCoverageBiasModel: Loaded model
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                
            # Create a new model instance
            model = cls(
                model_type="random_forest",
                bin_count=data.get('bin_count', 100),
                seed=data.get('seed', None),
                config=data.get('config', None),
                feature_cache_file=feature_cache_file
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
                "model_type": "random_forest"
            })
            
            # Ensure model_type is always random_forest
            model.parameters["model_type"] = "random_forest"
            
            logger.info(f"Loaded random forest coverage model from {filename}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a new model instance
            return cls(feature_cache_file=feature_cache_file) 