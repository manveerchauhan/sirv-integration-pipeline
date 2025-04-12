"""
Coverage bias modeling module for SIRV Integration Pipeline.

This module analyzes BAM files to model 5'-3' coverage bias for
transcript integration.

Note: This module includes the complete functionality for coverage bias modeling,
      including both the model itself and the learning algorithms. A deprecated
      version of the learning code was previously in models/coverage_learner.py.
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
import pickle
from collections import defaultdict
import re

from sirv_pipeline.utils import validate_files

# Set up logger
logger = logging.getLogger(__name__)

# Define full path to samtools
SAMTOOLS_PATH = "/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/SAMtools/1.21/bin/samtools"

class CoverageBiasModel:
    """
    Model to capture and simulate 5'-3' coverage bias in cDNA long reads.
    
    This model can represent several types of coverage bias:
    - 10x_cdna: 3' biased coverage common in 10X Chromium cDNA preparation
    - direct_rna: 5' biased coverage common in direct RNA sequencing
    - custom: Learned from actual aligned reads
    
    The model can be saved to and loaded from pickle files for reuse.
    """
    
    def __init__(self, model_type="10x_cdna", 
                 bin_count=100, 
                 smoothing_factor=0.05,
                 seed=None,
                 length_bins=10,
                 logger=None,
                 parameters=None):
        """
        Initialize the coverage bias model.
        
        Args:
            model_type: Type of model ('10x_cdna', 'direct_rna', or 'custom')
            bin_count: Number of bins to divide transcripts into
            smoothing_factor: Smoothing factor for kernel density estimation
            seed: Random seed for reproducibility
            length_bins: Number of bins to divide transcripts into for modeling.
            logger: Logger object for logging information and errors.
            parameters: Optional pre-defined parameters dictionary. If provided,
                        this will override the default model parameters.
        """
        # Initialize model parameters
        self.model_type = model_type
        self.bin_count = bin_count
        self.smoothing_factor = smoothing_factor
        self.seed = seed
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize distributions
        self.position_distribution = None  # Overall position distribution
        self.length_dependent_distributions = {}  # Length-stratified distributions
        self.length_bins = length_bins
        
        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        # Initialize parameters dict that will be used for plotting and sampling
        self.parameters = {}
            
        # Load from provided parameters if available
        if parameters is not None:
            self._load_from_parameters(parameters)
        # Otherwise use default model based on specified type
        elif model_type == "10x_cdna":
            self._init_default_10x_cdna_model()
        elif model_type == "direct_rna":
            self._init_default_direct_rna_model()
        elif model_type == "custom":
            # Initialize with uniform distribution
            # This should be updated by calling learn_from_bam later
            self._init_custom_uniform_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_from_parameters(self, parameters):
        """
        Load model from parameters dictionary.
        
        This handles both the case where parameters is a complete model dict
        (with 'parameters' and 'model_type' keys) and where it's just the
        inner parameters dict.
        
        Args:
            parameters: Dict containing model parameters
        """
        self.logger.info(f"Loading model from parameters dictionary")
        
        # Check if this is a complete model dict or just parameters
        if 'parameters' in parameters and isinstance(parameters['parameters'], dict):
            inner_params = parameters['parameters']
            # Update model type if available
            if 'model_type' in parameters:
                self.model_type = parameters['model_type']
                self.logger.info(f"Using model type from parameters: {self.model_type}")
        else:
            # Assume the dict is already the inner parameters
            inner_params = parameters
        
        # Store parameters
        if 'profile' in inner_params:
            self.parameters['profile'] = inner_params['profile']
        else:
            self.logger.warning("No profile found in parameters, using uniform")
            self.parameters['profile'] = np.ones(1000) / 1000
            
        if 'length_effect' in inner_params:
            self.parameters['length_effect'] = inner_params['length_effect']
        else:
            self.logger.warning("No length_effect found in parameters, using uniform")
            self.parameters['length_effect'] = {f'bin_{i}': 1.0 for i in range(5)}
        
        # Set up position distribution based on profile
        x = np.linspace(0, 1, len(self.parameters['profile']))
        self.position_distribution = (x, self.parameters['profile'])
        
        self.logger.info(f"Loaded model with profile length {len(self.parameters['profile'])} " +
                        f"and {len(self.parameters['length_effect'])} length bins")
    
    def _init_default_10x_cdna_model(self):
        """Initialize with default 10X Chromium cDNA bias model.
        
        This creates a model with strong 3' bias (more reads near 3' end),
        typical of 10x Genomics Chromium cDNA sequencing.
        """
        self.logger.info("Initializing default 10X Chromium cDNA bias model")
        
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
            
        # Initialize parameters for model use
        self.parameters = {
            "profile": y,  # Default profile (strongest bias)
            "length_effect": {f"bin_{i+1}": np.float64(params[1]/params[0]) for i, params in enumerate(beta_params)}
        }
    
    def _init_default_direct_rna_model(self):
        """Initialize with default direct RNA bias model.
        
        This creates a model with 5' bias (more reads near 5' end),
        typical of direct RNA sequencing.
        """
        self.logger.info("Initializing default direct RNA bias model")
        
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
            
        # Initialize parameters for model use
        self.parameters = {
            "profile": y,  # Default profile (strongest bias)
            "length_effect": {f"bin_{i+1}": np.float64(params[0]/params[1]) for i, params in enumerate(beta_params)}
        }
    
    def _init_custom_uniform_model(self):
        """Initialize with a uniform distribution as a starting point for custom models.
        
        This is used when a model will be learned from data later.
        """
        self.logger.info("Initializing uniform custom model")
        
        # Create a uniform distribution
        x = np.linspace(0, 1, 1000)
        y = np.ones_like(x)
        
        # Normalize
        y = y / np.sum(y)
        
        # Store as position distribution
        self.position_distribution = (x, y)
        
        # Define default length bins
        self.length_bins = [0, 500, 1000, 2000, 5000, np.inf]
        
        # Use uniform distribution for all length bins
        for i in range(5):
            self.length_dependent_distributions[i] = (x, y.copy())
            
        # Initialize parameters for model use
        self.parameters = {
            "profile": y,
            "length_effect": {f"bin_{i+1}": 1.0 for i in range(5)}
        }

    def learn_from_bam(self, bam_file, annotation_file, min_reads=100, length_bins=5):
        """Learn coverage bias from a BAM file.
        
        This analyzes read alignments to transcripts to build a custom
        coverage bias model.
        
        Args:
            bam_file (str): Path to BAM file
            annotation_file (str): Path to annotation file (GTF)
            min_reads (int): Minimum number of reads required for learning
            length_bins (int): Number of bins for transcript length distribution
            
        Returns:
            bool: True if learning was successful, False otherwise
        """
        import pysam
        import numpy as np
        import tempfile
        import os
        import subprocess
        from collections import defaultdict
        import re
        import traceback
        
        self.logger.info(f"Learning coverage bias from BAM file: {bam_file}")
        self.model_type = "custom"  # Mark as custom model
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix="coverage_model_")
        self.logger.info(f"Created temporary directory for BAM processing: {temp_dir}")
        
        try:
            # First, try to fix any issues with the BAM file by extracting and re-sorting
            fixed_bam = os.path.join(temp_dir, "fixed.bam")
            self.logger.info(f"Pre-processing BAM file to fix potential issues: {bam_file} -> {fixed_bam}")
            
            # Extract reads from original BAM and create a new, clean BAM file
            try:
                # Open input BAM file
                in_bam = pysam.AlignmentFile(bam_file, "rb")
                
                # Create output BAM file with same header
                out_bam = pysam.AlignmentFile(fixed_bam, "wb", header=in_bam.header)
                
                # Process and filter reads
                processed_reads = 0
                valid_reads = 0
                
                for read in in_bam.fetch(until_eof=True):
                    processed_reads += 1
                    
                    # Skip reads with serious issues
                    if read.query_name is None or read.cigarstring is None:
                        continue
                        
                    # Write read to output BAM
                    out_bam.write(read)
                    valid_reads += 1
                    
                    # Log progress
                    if processed_reads % 100000 == 0:
                        self.logger.info(f"Processed {processed_reads:,} reads, found {valid_reads:,} valid reads")
                
                # Close files
                in_bam.close()
                out_bam.close()
                
                self.logger.info(f"Processed {processed_reads:,} reads, found {valid_reads:,} valid reads")
                
                # Index the fixed BAM file
                self.logger.info(f"Indexing fixed BAM file")
                pysam.index(fixed_bam)
                
            except Exception as e:
                self.logger.error(f"Error pre-processing BAM file: {str(e)}")
                self.logger.error(traceback.format_exc())
                return False
            
            # Parse the annotation file
            self.logger.info(f"Parsing annotation file: {annotation_file}")
            try:
                transcript_info = self._parse_annotation(annotation_file)
                
                if not transcript_info:
                    self.logger.error(f"No transcript information found in {annotation_file}")
                    return False
                    
                self.logger.info(f"Found {len(transcript_info):,} transcripts in annotation file")
                
            except Exception as e:
                self.logger.error(f"Error parsing annotation file: {str(e)}")
                self.logger.error(traceback.format_exc())
                return False
            
            # Analyze read coverage
            self.logger.info(f"Analyzing read coverage")
            
            # Create analysis objects
            transcript_positions = defaultdict(list)
            
            # Open the fixed BAM file
            bam = pysam.AlignmentFile(fixed_bam, "rb")
            
            # Define function to process each read
            def process_read(read):
                # Skip secondary or supplementary alignments
                if read.is_secondary or read.is_supplementary:
                    return
                    
                # Extract transcript ID from reference name
                transcript_id = read.reference_name
                
                # Skip if not in transcript info
                if transcript_id not in transcript_info:
                    return
                    
                # Get transcript length
                transcript_length = transcript_info[transcript_id]["length"]
                
                # Get read position and convert to relative position
                read_pos = read.reference_start  # 0-based position
                relative_pos = read_pos / transcript_length
                
                # Store relative position
                transcript_positions[transcript_id].append(relative_pos)
            
            # Process each read
            processed_reads = 0
            for read in bam.fetch():
                process_read(read)
                processed_reads += 1
                
                # Log progress
                if processed_reads % 100000 == 0:
                    self.logger.info(f"Processed {processed_reads:,} reads")
            
            bam.close()
            
            self.logger.info(f"Processed {processed_reads:,} reads")
            self.logger.info(f"Found {len(transcript_positions):,} transcripts with read alignments")
            
            # Filter transcripts with insufficient reads
            filtered_transcripts = {
                tid: positions for tid, positions in transcript_positions.items()
                if len(positions) >= min_reads
            }
            
            self.logger.info(f"Found {len(filtered_transcripts):,} transcripts with at least {min_reads} reads")
            
            if not filtered_transcripts:
                self.logger.error(f"No transcripts with sufficient reads found")
                return False
            
            # Combine all positions to get overall distribution
            all_positions = []
            for positions in filtered_transcripts.values():
                all_positions.extend(positions)
            
            # Create position distribution
            x, y = self._create_position_density(all_positions)
            
            # Store as position distribution
            self.position_distribution = (x, y)
            
            # Group transcripts by length for length-dependent analysis
            if length_bins > 1:
                # Get lengths of all transcripts
                transcript_lengths = [transcript_info[tid]["length"] for tid in filtered_transcripts.keys()]
                
                # Create length bins
                length_bin_edges = np.percentile(transcript_lengths, np.linspace(0, 100, length_bins+1))
                self.length_bins = list(length_bin_edges)
                
                self.logger.info(f"Created {length_bins} length bins: {self.length_bins}")
                
                # Group transcripts by length bin
                transcript_by_bin = defaultdict(list)
                
                for tid, positions in filtered_transcripts.items():
                    transcript_length = transcript_info[tid]["length"]
                    
                    # Find the appropriate bin
                    bin_idx = 0
                    for i, edge in enumerate(self.length_bins[1:]):
                        if transcript_length > edge:
                            bin_idx = i + 1
                    
                    # Store in bin
                    transcript_by_bin[bin_idx].extend(positions)
                
                # Create distributions for each bin
                length_effect = {}
                
                for bin_idx, positions in transcript_by_bin.items():
                    if not positions:
                        continue
                        
                    # Create distribution
                    x_bin, y_bin = self._create_position_density(positions)
                    
                    # Store distribution
                    self.length_dependent_distributions[bin_idx] = (x_bin, y_bin)
                    
                    # Calculate bias measure
                    bin_5prime = np.sum(y_bin[:len(y_bin)//2])
                    bin_3prime = np.sum(y_bin[len(y_bin)//2:])
                    
                    # Use the 3'/5' ratio as the effect measure
                    # Higher values = more 3' biased
                    if bin_5prime > 0:
                        effect = bin_3prime / bin_5prime
                    else:
                        effect = 1.0
                        
                    length_effect[f"bin_{bin_idx+1}"] = np.float64(effect)
                    
                    self.logger.info(f"Length bin {bin_idx+1}: Effect={effect:.2f} " +
                                    f"(5'={bin_5prime:.2f}, 3'={bin_3prime:.2f})")
                
                # Store length effect
                self.parameters["length_effect"] = length_effect
            else:
                # Use one bin for all transcripts
                self.parameters["length_effect"] = {"bin_1": np.float64(1.0)}
            
            # Store profile for plotting
            self.parameters["profile"] = y
            
            # Set model type
            self.model_type = "custom"
            
            self.logger.info(f"Successfully learned coverage bias model from BAM file")
            return True
            
        except Exception as e:
            self.logger.error(f"Error learning coverage bias model: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _prepare_bam_file(self, bam_file):
        """
        Prepare BAM file for coverage analysis by checking, sorting, and indexing.
        
        Args:
            bam_file: Path to input BAM file
            
        Returns:
            str: Path to prepared BAM file, or None if preparation failed
        """
        import tempfile
        import subprocess
        import os
        import shutil
        
        if not os.path.exists(bam_file):
            logger.error(f"BAM file does not exist: {bam_file}")
            return None
            
        # Create a temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="coverage_model_")
        logger.info(f"Created temporary directory for BAM processing: {temp_dir}")
        
        try:
            # Check if BAM file is sorted
            is_sorted = False
            try:
                # Check BAM header for SO:coordinate
                header_cmd = f"samtools view -H {bam_file}"
                header_output = subprocess.check_output(header_cmd, shell=True, stderr=subprocess.STDOUT).decode('utf-8')
                is_sorted = "SO:coordinate" in header_output
            except subprocess.CalledProcessError:
                is_sorted = False
                
            # Sort BAM file if needed
            sorted_bam = os.path.join(temp_dir, "sorted.bam")
            if not is_sorted:
                logger.info(f"Sorting BAM file: {bam_file}")
                sort_cmd = f"samtools sort -o {sorted_bam} {bam_file}"
                try:
                    subprocess.check_call(sort_cmd, shell=True, stderr=subprocess.STDOUT)
                    logger.info(f"Successfully sorted BAM file to: {sorted_bam}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to sort BAM file: {str(e)}")
                    shutil.rmtree(temp_dir)
                    return None
            else:
                # If already sorted, just copy the file
                shutil.copy(bam_file, sorted_bam)
                logger.info(f"BAM file already sorted: {bam_file}")
                
            # Index the sorted BAM file
            logger.info(f"Indexing BAM file: {sorted_bam}")
            index_cmd = f"samtools index {sorted_bam}"
            try:
                subprocess.check_call(index_cmd, shell=True, stderr=subprocess.STDOUT)
                logger.info(f"Successfully indexed BAM file: {sorted_bam}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to index BAM file: {str(e)}")
                logger.warning("Proceeding without index, but performance may be affected")
                
            return sorted_bam
            
        except Exception as e:
            logger.error(f"Error preparing BAM file: {str(e)}")
            shutil.rmtree(temp_dir)
            return None

    def _create_position_density(self, positions):
        """
        Create a position density distribution from a list of positions.
        
        Args:
            positions: List of normalized positions (0-1)
            
        Returns:
            tuple: (x_values, y_values) for the density distribution
        """
        if not positions:
            logger.warning("No positions provided for density estimation. Using uniform distribution.")
            x = np.linspace(0, 1, 1000)
            y = np.ones_like(x) / len(x)
            return (x, y)
            
        positions = np.array(positions)
        
        # Remove any NaN or infinite values
        positions = positions[~np.isnan(positions) & ~np.isinf(positions)]
        
        if len(positions) < 2:
            logger.warning("Insufficient positions for density estimation. Using uniform distribution.")
            x = np.linspace(0, 1, 1000)
            y = np.ones_like(x) / len(x)
            return (x, y)
            
        try:
            # Create kernel density estimate
            kde = stats.gaussian_kde(positions, bw_method=self.smoothing_factor)
            
            # Evaluate on a grid
            x = np.linspace(0, 1, 1000)
            y = kde(x)
            
            # Normalize
            y = y / np.sum(y)
            
            return (x, y)
            
        except Exception as e:
            logger.warning(f"Error creating density estimate: {str(e)}. Using uniform distribution.")
            x = np.linspace(0, 1, 1000)
            y = np.ones_like(x) / len(x)
            return (x, y)
    
    def _sample_from_distribution(self, distribution):
        """
        Sample a value from a probability distribution.
        
        Args:
            distribution: Tuple of (x_values, probabilities)
            
        Returns:
            float: Sampled value from the distribution
        """
        x, y = distribution
        return np.random.choice(x, p=y/np.sum(y))
    
    def _parse_annotation(self, annotation_file):
        """
        Parse annotation file to get transcript information.
        
        Args:
            annotation_file: Path to GTF/GFF annotation file
            
        Returns:
            dict: Dictionary of transcript information
        """
        transcripts = {}
        
        try:
            with open(annotation_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    fields = line.strip().split('\t')
                    if len(fields) < 9:
                        continue
                        
                    if fields[2].lower() in ('transcript', 'mrna'):
                        # Try to extract transcript ID from attributes field
                        attributes = fields[8]
                        transcript_id = None
                        
                        # Try different formats
                        # Format 1: transcript_id "ENST00000456328.2";
                        if 'transcript_id' in attributes:
                            try:
                                transcript_id = attributes.split('transcript_id')[1].split(';')[0].strip().strip('"\'')
                            except:
                                pass
                        
                        # Format 2: ID=transcript:ENST00000456328.2;
                        if transcript_id is None and 'ID=' in attributes:
                            try:
                                id_part = [p for p in attributes.split(';') if p.strip().startswith('ID=')]
                                if id_part:
                                    transcript_id = id_part[0].split('=')[1].strip().strip('"\'')
                                    if ':' in transcript_id:
                                        transcript_id = transcript_id.split(':')[1]
                            except:
                                pass
                        
                        # Format 3: Parent=gene:ENSG00000223972.5;ID=ENST00000456328.2;
                        if transcript_id is None and 'ID=' in attributes:
                            try:
                                id_part = [p for p in attributes.split(';') if p.strip().startswith('ID=')]
                                if id_part:
                                    transcript_id = id_part[0].split('=')[1].strip().strip('"\'')
                            except:
                                pass
                                
                        # Simple format: ID=ENST00000456328.2
                        if transcript_id is None and attributes.startswith('ID='):
                            try:
                                transcript_id = attributes.split('=')[1].split(';')[0].strip().strip('"\'')
                            except:
                                pass
                        
                        # If we found a transcript ID and it's valid
                        if transcript_id:
                            # Get strand
                            strand = fields[6]
                            
                            # Get length (end - start + 1)
                            try:
                                start = int(fields[3])
                                end = int(fields[4])
                                length = end - start + 1
                            except:
                                # If we can't calculate length, skip this transcript
                                continue
                            
                            transcripts[transcript_id] = {'strand': strand, 'length': length}
        except Exception as e:
            logger.warning(f"Error parsing annotation file: {str(e)}")
            
        if not transcripts:
            logger.warning("No transcripts found in annotation file. Using fallback parsing method.")
            
            try:
                # Fallback to simple parsing
                with open(annotation_file, 'r') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue
                        fields = line.strip().split('\t')
                        if len(fields) < 8:
                            continue
                            
                        # Just use the first column as ID and assume it's a transcript
                        transcript_id = fields[0]
                        strand = fields[6] if len(fields) > 6 else '+'
                        try:
                            start = int(fields[3]) if len(fields) > 3 else 1
                            end = int(fields[4]) if len(fields) > 4 else 1000
                            length = end - start + 1
                        except:
                            length = 1000  # Default length
                        
                        transcripts[transcript_id] = {'strand': strand, 'length': length}
            except Exception as e:
                logger.error(f"Fallback parsing also failed: {str(e)}")
        
        logger.info(f"Parsed {len(transcripts)} transcripts from annotation")
        return transcripts

    def save(self, filename):
        """Save the coverage bias model to a file.
        
        Args:
            filename (str): Path to save the model
        """
        # Convert numpy arrays to lists for serialization
        serializable_params = {}
        for key, value in self.parameters.items():
            if isinstance(value, np.ndarray):
                serializable_params[key] = value.tolist()
            else:
                serializable_params[key] = value
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save model to file
        model_data = {
            'model_type': self.model_type,
            'parameters': serializable_params,
            'bin_count': self.bin_count,
            'seed': self.seed
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Coverage bias model saved to {filename}")

    def load(self, filename):
        """Load the coverage bias model from a file.
        
        Args:
            filename (str): Path to load the model from
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            # Set model attributes from loaded data
            self.model_type = model_data.get('model_type', self.model_type)
            self.bin_count = model_data.get('bin_count', self.bin_count)
            self.seed = model_data.get('seed', self.seed)
            
            # Convert lists back to numpy arrays if needed
            loaded_params = model_data.get('parameters', {})
            self.parameters = {}
            for key, value in loaded_params.items():
                if isinstance(value, list):
                    self.parameters[key] = np.array(value)
                else:
                    self.parameters[key] = value
            
            # Set seed if it was saved
            if self.seed is not None:
                np.random.seed(self.seed)
            
            self.logger.info(f"Successfully loaded coverage model from {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading coverage model: {str(e)}")
            # Initialize default model as fallback
            if self.model_type == "10x_cdna":
                self._init_default_10x_cdna_model()
            elif self.model_type == "direct_rna":
                self._init_default_direct_rna_model()
            return False

    def apply_to_sequence(self, sequence, quality, target_length=None):
        """Apply coverage bias model to a sequence.
        
        Args:
            sequence (str): Input sequence
            quality (str): Quality string for the sequence
            target_length (int, optional): Target length for the output sequence
            
        Returns:
            tuple: (biased_sequence, biased_quality)
        """
        try:
            # If no target length specified, use original length
            if target_length is None or target_length <= 0:
                target_length = len(sequence)
                
            # Cap target length to original sequence length
            target_length = min(target_length, len(sequence))
            
            # Sample position from bias model
            if "profile" in self.parameters:
                # Use the model's profile to determine read start or end position
                profile = self.parameters["profile"]
                x = np.linspace(0, 1, len(profile))
                
                # Sample relative position (0-1)
                rel_pos = np.random.choice(x, p=profile/np.sum(profile))
                
                # Convert to sequence position
                seq_len = len(sequence)
                
                # For 3' bias models (10x_cdna), higher values of rel_pos mean more 3' bias
                # For 5' bias models (direct_rna), lower values of rel_pos mean more 5' bias
                if self.model_type == "10x_cdna":
                    # 3' bias - higher rel_pos means start closer to 3' end
                    start_pos = int((seq_len - target_length) * rel_pos)
                else:
                    # 5' bias or custom - lower rel_pos means start closer to 5' end
                    start_pos = int((seq_len - target_length) * (1 - rel_pos))
                
                # Ensure start_pos is within bounds
                start_pos = max(0, min(start_pos, seq_len - target_length))
            else:
                # Fallback to random sampling if no profile
                max_start = max(0, len(sequence) - target_length)
                start_pos = np.random.randint(0, max_start + 1)
            
            # Extract subsequence
            end_pos = start_pos + target_length
            biased_seq = sequence[start_pos:end_pos]
            
            # Extract corresponding quality values
            if quality and len(quality) >= end_pos:
                biased_qual = quality[start_pos:end_pos]
            else:
                # Generate placeholder quality if not available
                biased_qual = 'I' * len(biased_seq)
            
            return biased_seq, biased_qual
            
        except Exception as e:
            self.logger.error(f"Error applying coverage bias to sequence: {str(e)}")
            # Return original sequence as fallback
            return sequence[:target_length], quality[:target_length] if quality else 'I' * target_length

    def plot_distributions(self, output_file=None):
        """Plot the coverage bias distribution.
        
        Args:
            output_file (str): Path to save the plot (optional)
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot coverage profile
        x = np.linspace(0, 1, len(self.parameters["profile"]))
        ax1.plot(x, self.parameters["profile"], linewidth=2.5)
        ax1.set_xlabel("Relative position along transcript (5' → 3')", fontsize=12)
        ax1.set_ylabel("Coverage bias factor", fontsize=12)
        ax1.set_title("Coverage bias across transcript length", fontsize=14)
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
        y_max = max(self.parameters["profile"])
        y_pos = 0.8 * y_max
        bias_type = "3' biased" if np.argmax(self.parameters["profile"]) > len(self.parameters["profile"])/2 else "5' biased"
        ax1.text(0.5, 0.9, f"This model is {bias_type}", transform=ax1.transAxes, 
                 fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        ax1.legend(loc='upper center')
        
        # Plot length effect if available
        if "length_effect" in self.parameters and self.parameters["length_effect"]:
            length_effect = self.parameters["length_effect"]
            groups = list(length_effect.keys())
            values = [length_effect[g] for g in groups]
            
            # Bar plot for length effect
            bar_positions = np.arange(len(groups))
            bars = ax2.bar(bar_positions, values, color='steelblue')
            ax2.set_xticks(bar_positions)
            
            # Create more informative x-tick labels from bin names
            readable_labels = []
            for g in groups:
                if g.startswith('bin_'):
                    bin_num = int(g.replace('bin_', ''))
                    # Show bin numbers as transcript length groups
                    readable_labels.append(f"Group {bin_num}")
                else:
                    readable_labels.append(g)
            
            ax2.set_xticklabels(readable_labels, rotation=45)
            ax2.set_xlabel("Transcript length group", fontsize=12)
            ax2.set_ylabel("Length effect strength", fontsize=12)
            ax2.set_title("Effect of transcript length on coverage bias", fontsize=14)
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
        model_desc = {
            "10x_cdna": "10x Chromium cDNA (typically 3' biased)",
            "direct_rna": "Direct RNA sequencing (typically 5' biased)",
            "custom": "Custom coverage model learned from data"
        }.get(self.model_type, self.model_type)
        
        # Add badge to indicate if it's a default or custom model
        is_default = self.model_type in ["10x_cdna", "direct_rna"]
        model_status = "DEFAULT MODEL" if is_default else "CUSTOM MODEL"
        status_color = "orange" if is_default else "green"
        
        fig.suptitle(f"Coverage Bias Model: {model_desc}", fontsize=16)
        
        # Add a badge showing if it's a default or custom model
        fig.text(0.01, 0.97, model_status, 
                fontsize=10, ha='left', va='top', weight='bold',
                bbox=dict(facecolor=status_color, alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Add an explanatory text box at the bottom
        explanation = (
            "EXPLANATION: This plot shows how read coverage varies across transcript length.\n"
            "LEFT: Values >1 indicate over-representation, <1 indicate under-representation at that position.\n"
            "RIGHT: How transcript length affects the strength of the coverage bias."
        )
        fig.text(0.5, 0.01, explanation, ha='center', va='bottom', fontsize=11,
                bbox=dict(facecolor='lightyellow', alpha=0.9, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save figure if output file is specified
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved coverage bias plot to {output_file}")
        
        return fig

def model_transcript_coverage(
    bam_file: str,
    gtf_file: str,
    output_csv: str,
    num_bins: int = 100
) -> pd.DataFrame:
    """
    Legacy function to model transcript coverage from BAM file.
    
    Args:
        bam_file: Path to BAM file with aligned reads
        gtf_file: Path to GTF/GFF annotation file
        output_csv: Path to output CSV file
        num_bins: Number of transcript length bins
        
    Returns:
        DataFrame with coverage model information
    """
    # Extract transcript coordinates
    logger.info(f"Extracting transcript coordinates from {gtf_file}")
    transcript_coords = extract_transcript_coordinates(gtf_file)
    
    # Calculate coverage for each transcript
    coverage_dict = {}
    for transcript_id, coords in transcript_coords.items():
        try:
            coverage = calculate_transcript_coverage(
                bam_file, 
                coords['chrom'],
                coords['start'],
                coords['end'],
                num_bins
            )
            coverage_dict[transcript_id] = coverage
        except Exception as e:
            logger.warning(f"Error calculating coverage for {transcript_id}: {e}")
    
    # Convert to DataFrame
    df = _convert_model_to_dataframe(coverage_dict, num_bins)
    
    # Save to CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved coverage model to {output_csv}")
    
    return df

def _convert_model_to_dataframe(model, num_bins):
    """Convert coverage model dictionary to DataFrame."""
    df_rows = []
    
    for transcript_id, coverage in model.items():
        if len(coverage) != num_bins:
            # Skip transcripts with wrong bin count
            continue
            
        # Create a row for each transcript with all bins
        row = {"transcript_id": transcript_id}
        for i, value in enumerate(coverage):
            row[f"bin_{i+1}"] = value
            
        df_rows.append(row)
    
    return pd.DataFrame(df_rows)

def extract_transcript_coordinates(gtf_file: str) -> Dict[str, Dict[str, int]]:
    """Extract transcript coordinates from GTF file."""
    result = {}
    
    with open(gtf_file, 'r') as f:
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
                        'end': end
                    }
    
    return result

def calculate_transcript_coverage(
    bam_file: str,
    chrom: str,
    start: int,
    end: int,
    num_bins: int = 100
) -> List[float]:
    """Calculate normalized coverage for a transcript region."""
    # Get raw coverage
    coverage = get_coverage_from_bam(bam_file, chrom, start, end)
    
    # Bin the coverage
    binned_coverage = bin_coverage(coverage, num_bins)
    
    return binned_coverage

def get_coverage_from_bam(
    bam_file: str,
    chrom: str,
    start: int,
    end: int
) -> List[int]:
    """
    Get coverage for a genomic region from BAM file.
    
    Args:
        bam_file: Path to sorted and indexed BAM file
        chrom: Chromosome/contig name
        start: Start position (1-based, inclusive)
        end: End position (1-based, inclusive)
        
    Returns:
        List of coverage values for each position
    """
    import pysam
    
    # Initialize coverage array
    length = end - start + 1
    coverage = [0] * length
    
    # Read BAM file
    try:
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            # Fetch reads in the region
            for read in bam.fetch(chrom, start-1, end):  # pysam uses 0-based coordinates
                # Get read alignment start and end
                read_start = max(0, read.reference_start - (start-1))
                read_end = min(length, read.reference_end - (start-1))
                
                # Increment coverage
                for i in range(read_start, read_end):
                    if 0 <= i < length:
                        coverage[i] += 1
    except Exception as e:
        logger.warning(f"Error reading BAM file {bam_file}: {e}")
        
    return coverage

def bin_coverage(coverage: List[int], num_bins: int) -> List[float]:
    """
    Bin coverage values into a fixed number of bins.
    
    Args:
        coverage: List of coverage values for each position
        num_bins: Number of bins
        
    Returns:
        List of binned coverage values
    """
    import numpy as np
    
    # Handle empty or too small coverage
    if not coverage or len(coverage) < num_bins:
        return [0.0] * num_bins
        
    # Calculate bin size
    bin_size = len(coverage) / num_bins
    
    # Initialize bins
    binned = []
    
    # Aggregate coverage into bins
    for i in range(num_bins):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size)
        
        # Handle edge case for last bin
        if i == num_bins - 1:
            end_idx = len(coverage)
            
        # Calculate mean coverage for this bin
        bin_coverage = np.mean(coverage[start_idx:end_idx]) if end_idx > start_idx else 0
        binned.append(float(bin_coverage))
    
    # Normalize to sum to 1.0
    total = sum(binned)
    if total > 0:
        binned = [v / total for v in binned]
    
    return binned 

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
    Create a coverage bias model from FASTQ or BAM file.
    
    Args:
        fastq_file: Path to FASTQ file with reads
        bam_file: Path to BAM file with aligned reads
        annotation_file: Path to GTF/GFF annotation file
        model_type: Type of model ('10x_cdna', 'direct_rna', or 'custom')
        sample_size: Number of reads to sample for analysis
        min_reads: Minimum number of reads for a transcript to be included
        length_bins: Number of transcript length bins
        seed: Random seed for reproducibility
        
    Returns:
        CoverageBiasModel: Initialized coverage bias model
    """
    # Create initial model with default parameters
    model = CoverageBiasModel(model_type=model_type, seed=seed)
    
    # If BAM file is provided, learn from BAM
    if bam_file and annotation_file:
        logger.info(f"Learning coverage bias from BAM file: {bam_file}")
        model.learn_from_bam(
            bam_file=bam_file,
            annotation_file=annotation_file,
            min_reads=min_reads,
            length_bins=length_bins
        )
    # If FASTQ file is provided but no BAM
    elif fastq_file and not bam_file:
        logger.info(f"Using default {model_type} coverage bias model")
        # We'll just use the default model initialized in the constructor
        # Could add sampling from FASTQ in the future if needed
    else:
        logger.info(f"Using default {model_type} coverage bias model")
        # Model already has defaults initialized
    
    return model 

class ReadLengthSampler:
    """
    Class for sampling read lengths based on empirical distributions.
    """
    
    def __init__(self, lengths=None, min_length=200, max_length=15000, seed=None):
        """
        Initialize read length sampler.
        
        Args:
            lengths: Optional list of read lengths to sample from
            min_length: Minimum read length to sample
            max_length: Maximum read length to sample
            seed: Random seed for reproducibility
        """
        self.min_length = min_length
        self.max_length = max_length
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # If lengths are provided, use empirical distribution
        if lengths is not None and len(lengths) > 0:
            # Filter out invalid lengths
            valid_lengths = [l for l in lengths if min_length <= l <= max_length]
            
            if valid_lengths:
                self.lengths = valid_lengths
            else:
                logger.warning(f"No valid lengths found in provided list. Using default range.")
                self.lengths = None
        else:
            self.lengths = None
    
    def sample(self):
        """
        Sample a read length.
        
        Returns:
            int: Sampled read length
        """
        if self.lengths:
            # Sample from empirical distribution
            return np.random.choice(self.lengths)
        else:
            # Sample from log-normal distribution (common distribution for read lengths)
            # These parameters approximate ONT read length distributions
            mu, sigma = 8.5, 0.5  # Natural log parameters
            length = int(np.random.lognormal(mu, sigma))
            
            # Ensure length is within bounds
            return max(self.min_length, min(length, self.max_length))

def sample_read_lengths(fastq_file, sample_size=1000):
    """
    Sample read lengths from a FASTQ file.
    
    Args:
        fastq_file: Path to FASTQ file
        sample_size: Number of reads to sample
        
    Returns:
        List of read lengths
    """
    from Bio import SeqIO
    import random
    
    # Check if file exists
    if not os.path.exists(fastq_file):
        logger.error(f"FASTQ file not found: {fastq_file}")
        return []
        
    try:
        # Count records
        with open(fastq_file, 'r') as f:
            record_count = sum(1 for _ in SeqIO.parse(f, 'fastq'))
            
        # If sample size is larger than record count, adjust
        if sample_size > record_count:
            sample_size = record_count
            
        # Randomly sample reads
        lengths = []
        with open(fastq_file, 'r') as f:
            records = list(SeqIO.parse(f, 'fastq'))
            samples = random.sample(records, sample_size)
            lengths = [len(record.seq) for record in samples]
            
        return lengths
    except Exception as e:
        logger.error(f"Error sampling read lengths: {str(e)}")
        return [] 