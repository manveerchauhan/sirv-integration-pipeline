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

    def learn_from_bam(self, bam_file, annotation_file, min_reads=100, length_bins=5):
        """Learn coverage bias from a BAM file.
        
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
                        
                    # Write to new BAM file
                    out_bam.write(read)
                    valid_reads += 1
                    
                    # Give progress updates for large files
                    if processed_reads % 1000000 == 0:
                        self.logger.info(f"Processed {processed_reads} reads, kept {valid_reads} valid reads")
                
                in_bam.close()
                out_bam.close()
                
                self.logger.info(f"Processed {processed_reads} reads, kept {valid_reads} valid reads")
                
                # Sort the fixed BAM file
                sorted_bam = os.path.join(temp_dir, "sorted.bam")
                self.logger.info(f"Sorting fixed BAM file: {fixed_bam} -> {sorted_bam}")
                
                # Use pysam for sorting
                pysam.sort("-o", sorted_bam, fixed_bam)
                self.logger.info(f"Successfully sorted BAM file to: {sorted_bam}")
                
                # Try to index the sorted BAM file
                self.logger.info(f"Indexing BAM file: {sorted_bam}")
                try:
                    pysam.index(sorted_bam)
                    self.logger.info(f"Successfully indexed BAM file: {sorted_bam}")
                except Exception as e:
                    self.logger.warning(f"Could not index BAM file: {str(e)}")
                    self.logger.warning("Will continue without index, but performance may be affected")
            except Exception as e:
                self.logger.error(f"Error pre-processing BAM file: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.logger.warning("Using original BAM file")
                
                # Fall back to simple sorting of original BAM
                sorted_bam = os.path.join(temp_dir, "sorted.bam")
                self.logger.info(f"Sorting original BAM file: {bam_file} -> {sorted_bam}")
                
                try:
                    pysam.sort("-o", sorted_bam, bam_file)
                    self.logger.info(f"Successfully sorted BAM file to: {sorted_bam}")
                except Exception as e:
                    self.logger.error(f"Could not sort BAM file: {str(e)}")
                    self.logger.error("Will use original BAM file directly")
                    sorted_bam = bam_file
            
            # Now process the GTF file to get transcript information
            self.logger.info(f"Parsing transcripts from annotation: {annotation_file}")
            transcript_lengths = {}
            transcript_names = {}  # Map for alternative names
            
            # Try parsing with gffutils first
            try:
                import gffutils
                import tempfile
                
                # Create an in-memory database
                db_file = os.path.join(temp_dir, "gtf.db")
                
                # Try creating the gffutils database
                try:
                    db = gffutils.create_db(
                        annotation_file,
                        dbfn=db_file,
                        force=True,
                        keep_order=True,
                        merge_strategy='merge',
                        sort_attribute_values=True,
                        disable_infer_genes=True,
                        disable_infer_transcripts=True
                    )
                    
                    # Get all transcripts
                    for tx in db.features_of_type('transcript'):
                        tx_id = tx.id
                        if 'transcript_id' in tx.attributes:
                            tx_id = tx.attributes['transcript_id'][0]
                        
                        # Calculate length
                        length = tx.end - tx.start + 1
                        transcript_lengths[tx_id] = length
                        
                        # Add alternative names
                        if '.' in tx_id:
                            base_id = tx_id.split('.')[0]
                            transcript_lengths[base_id] = length
                            transcript_names[base_id] = tx_id
                        
                        # Also store chromosome name if different
                        if tx.seqid != tx_id:
                            transcript_lengths[tx.seqid] = length
                            transcript_names[tx.seqid] = tx_id
                            
                    self.logger.info(f"Parsed {len(transcript_lengths)} transcripts using gffutils")
                except Exception as e:
                    self.logger.warning(f"Could not use gffutils to parse GTF: {str(e)}")
                    raise
            except Exception:
                # Fall back to manual parsing
                self.logger.info("Falling back to manual GTF parsing")
                
                with open(annotation_file) as f:
                    for line in f:
                        if line.startswith('#'):
                            continue
                        
                        fields = line.strip().split('\t')
                        if len(fields) < 9:
                            continue
                        
                        feature_type = fields[2]
                        if feature_type != 'transcript' and feature_type != 'exon':
                            continue
                        
                        # Extract transcript ID from attributes
                        attr_str = fields[8]
                        chr_name = fields[0]
                        tx_id = None
                        
                        # Try different formats of transcript_id attribute
                        tx_id_match = re.search(r'transcript_id "([^"]+)"', attr_str)
                        if tx_id_match:
                            tx_id = tx_id_match.group(1)
                        else:
                            tx_id_match = re.search(r'transcript_id=([^;]+)', attr_str)
                            if tx_id_match:
                                tx_id = tx_id_match.group(1).strip()
                            else:
                                # Try gene_id as a fallback
                                gene_id_match = re.search(r'gene_id "([^"]+)"', attr_str)
                                if gene_id_match:
                                    tx_id = gene_id_match.group(1)
                                else:
                                    # Last resort: use chromosome name as ID
                                    tx_id = chr_name
                        
                        if feature_type == 'transcript':
                            start = int(fields[3]) - 1  # 0-based
                            end = int(fields[4])
                            length = end - start
                            
                            transcript_lengths[tx_id] = length
                            
                            # Store additional reference names for better matching
                            # 1. Without version number (e.g., ENST00000456328.2 -> ENST00000456328)
                            if '.' in tx_id:
                                base_id = tx_id.split('.')[0]
                                transcript_lengths[base_id] = length
                                transcript_names[base_id] = tx_id
                            
                            # 2. Store the chromosome/reference name too
                            if chr_name != tx_id:
                                transcript_lengths[chr_name] = length
                                transcript_names[chr_name] = tx_id
                            
                            # 3. Store with and without "transcript:" prefix
                            if tx_id.startswith("transcript:"):
                                simple_id = tx_id.replace("transcript:", "")
                                transcript_lengths[simple_id] = length
                                transcript_names[simple_id] = tx_id
                            else:
                                prefixed_id = f"transcript:{tx_id}"
                                transcript_lengths[prefixed_id] = length
                                transcript_names[prefixed_id] = tx_id
            
                self.logger.info(f"Parsed {len(transcript_lengths)} transcripts from GTF")
            
            # Now open the BAM file and start processing reads
            self.logger.info(f"Processing reads from BAM file: {sorted_bam}")
            
            # Process BAM file to get coverage data
            coverage_data = defaultdict(lambda: defaultdict(int))
            transcript_counts = defaultdict(int)
            position_counts = defaultdict(int)
            
            # Reference matching stats
            found_refs = set()
            missing_refs = set()
            
            # Open BAM file - try with index first, fall back to no index
            try:
                bam = pysam.AlignmentFile(sorted_bam, "rb")
                
                # Get BAM references 
                bam_refs = list(bam.references) if bam.references else []
                
                # Log info about BAM references
                self.logger.info(f"BAM file has {len(bam_refs)} reference sequences")
                if bam_refs:
                    self.logger.info(f"First 5 BAM references: {bam_refs[:5]}")
                
                # Log info about transcript lengths
                self.logger.info(f"We have {len(transcript_lengths)} transcript lengths")
                if transcript_lengths:
                    tx_sample = list(transcript_lengths.keys())[:5]
                    self.logger.info(f"First 5 transcript IDs: {tx_sample}")
                
                # Build a mapping for reference name variants
                ref_name_map = {}
                
                # First, add exact mapping for BAM references
                for ref in bam_refs:
                    ref_name_map[ref] = ref
                    
                    # Add variant mappings
                    # Without version number
                    if '.' in ref:
                        base_ref = ref.split('.')[0]
                        ref_name_map[base_ref] = ref
                    
                    # Without ENST prefix if present
                    if ref.startswith('ENST'):
                        ref_name_map[ref[4:]] = ref
                    
                    # With and without "transcript:" prefix
                    if ref.startswith("transcript:"):
                        simple_ref = ref.replace("transcript:", "")
                        ref_name_map[simple_ref] = ref
                    else:
                        prefixed_ref = f"transcript:{ref}"
                        ref_name_map[prefixed_ref] = ref
                
                # Next, add all transcript IDs from GTF for reverse lookup
                for tx_id in transcript_lengths.keys():
                    if tx_id not in ref_name_map:
                        ref_name_map[tx_id] = tx_id
                
                # Count total reads
                try:
                    total_reads = bam.count()
                    self.logger.info(f"BAM file has {total_reads} total reads")
                except:
                    self.logger.warning("Could not count total reads in BAM")
                    total_reads = 0
                
                # Process reads
                processed_reads = 0
                skipped_reads = 0
                total_read_positions = 0
                reads_with_tx = 0
                
                # Try to use fetch with index first
                has_index = os.path.exists(sorted_bam + '.bai')
                
                # Function to process a read
                def process_read(read):
                    nonlocal processed_reads, skipped_reads, total_read_positions, reads_with_tx
                    
                    processed_reads += 1
                    
                    # Skip unmapped reads
                    if read.is_unmapped:
                        skipped_reads += 1
                        return
                        
                    # Skip secondary/supplementary alignments
                    if read.is_secondary or read.is_supplementary:
                        skipped_reads += 1
                        return
                    
                    # Get reference name and try to match it to our transcript list
                    ref_name = read.reference_name
                    if not ref_name or ref_name == '*':
                        skipped_reads += 1
                        return
                        
                    tx_id = None
                    
                    # Try direct match
                    if ref_name in transcript_lengths:
                        tx_id = ref_name
                        found_refs.add(ref_name)
                    elif ref_name in transcript_names:
                        tx_id = transcript_names[ref_name]
                        found_refs.add(tx_id)
                    else:
                        # Try lookup through our reference map
                        for variant, orig_ref in ref_name_map.items():
                            if variant in transcript_lengths:
                                tx_id = variant
                                found_refs.add(tx_id)
                                break
                        
                        # If still not found, try removing version numbers and prefixes
                        if not tx_id:
                            # Without version
                            if '.' in ref_name:
                                base_ref = ref_name.split('.')[0]
                                if base_ref in transcript_lengths:
                                    tx_id = base_ref
                                    found_refs.add(base_ref)
                            
                            # With/without "transcript:" prefix
                            if not tx_id:
                                if ref_name.startswith("transcript:"):
                                    simple_ref = ref_name.replace("transcript:", "")
                                    if simple_ref in transcript_lengths:
                                        tx_id = simple_ref
                                        found_refs.add(simple_ref)
                                else:
                                    prefixed_ref = f"transcript:{ref_name}"
                                    if prefixed_ref in transcript_lengths:
                                        tx_id = prefixed_ref
                                        found_refs.add(prefixed_ref)
                    
                    if not tx_id:
                        if ref_name not in missing_refs:
                            missing_refs.add(ref_name)
                            if len(missing_refs) <= 10:
                                self.logger.debug(f"Reference not found in transcript list: {ref_name}")
                        skipped_reads += 1
                        return
                    
                    # We found a matching transcript!
                    reads_with_tx += 1
                    transcript_counts[tx_id] += 1
                    
                    # Get transcript length
                    tx_len = transcript_lengths[tx_id]
                    
                    # Process read positions
                    for pos in range(read.reference_start, read.reference_end):
                        rel_pos = pos / tx_len  # Position as fraction of transcript length
                        bin_idx = int(rel_pos * self.bin_count)
                        
                        # Ensure bin_idx is within range
                        if bin_idx < 0:
                            bin_idx = 0
                        elif bin_idx >= self.bin_count:
                            bin_idx = self.bin_count - 1
                        
                        coverage_data[tx_id][bin_idx] += 1
                        position_counts[tx_id] += 1
                        total_read_positions += 1
                
                # Process reads - try with index first, fall back to until_eof
                if has_index:
                    try:
                        # Process each reference that matches a transcript
                        refs_to_process = []
                        for ref in bam_refs:
                            # Check if this reference exists in our transcript map
                            if ref in transcript_lengths or ref in transcript_names:
                                refs_to_process.append(ref)
                            elif ref in ref_name_map and ref_name_map[ref] in transcript_lengths:
                                refs_to_process.append(ref)
                        
                        if refs_to_process:
                            self.logger.info(f"Processing reads for {len(refs_to_process)} matching references")
                            
                            for ref in refs_to_process:
                                try:
                                    for read in bam.fetch(ref):
                                        process_read(read)
                                except Exception as e:
                                    self.logger.warning(f"Error fetching reads for reference {ref}: {str(e)}")
                        else:
                            # No matching references found, process all reads
                            self.logger.warning("No matching references found, processing all reads")
                            for read in bam.fetch(until_eof=True):
                                process_read(read)
                    except Exception as e:
                        self.logger.warning(f"Error using indexed fetch: {str(e)}")
                        self.logger.warning("Falling back to processing all reads")
                        # Fall back to processing all reads
                        for read in bam.fetch(until_eof=True):
                            process_read(read)
                else:
                    # No index, process all reads
                    self.logger.info("No BAM index, processing all reads")
                    for read in bam.fetch(until_eof=True):
                        process_read(read)
                        
                    # Give progress updates for large files
                    if processed_reads % 1000000 == 0:
                        self.logger.info(f"Processed {processed_reads} reads, found {reads_with_tx} with transcript matches")
                
                bam.close()
                
                # Log statistics
                self.logger.info(f"Processed {processed_reads} reads")
                self.logger.info(f"Skipped {skipped_reads} reads")
                self.logger.info(f"Found {len(found_refs)} matching transcripts")
                self.logger.info(f"Missing {len(missing_refs)} reference sequences")
                self.logger.info(f"Retained {total_read_positions} valid read positions")
                self.logger.info(f"Reads with transcript matches: {reads_with_tx}")
                
                # Check if we have enough data
                if total_read_positions < min_reads:
                    self.logger.error(f"Not enough read positions for learning: {total_read_positions} < {min_reads}")
                    
                    # If we have no reads but references were found, try using default coverage
                    if len(found_refs) > 0:
                        self.logger.warning("References were found but not enough read positions. Using a simple model with minimal data.")
                        
                        # Create a very simple coverage profile with slight 3' bias (typical for cDNA)
                        coverage_profile = np.linspace(0.8, 1.2, self.bin_count)
                        
                        # Create a uniform length effect
                        length_effect = {'default': 1.0}
                        
                        # Update model parameters
                        self.parameters = {
                            "profile": coverage_profile,
                            "length_effect": length_effect
                        }
                        
                        self.logger.info("Created simple coverage bias model with minimal data")
                        return True
                    
                    return False
                
                # Calculate coverage profile
                coverage_profile = np.zeros(self.bin_count)
                
                # Only use transcripts with sufficient coverage
                tx_with_coverage = [tx for tx, count in transcript_counts.items() if count >= 5]
                self.logger.info(f"Using {len(tx_with_coverage)} transcripts for coverage profile")
                
                if not tx_with_coverage:
                    self.logger.error("No transcripts with sufficient coverage")
                    return False
                
                # Calculate normalized coverage for each transcript
                for tx_id in tx_with_coverage:
                    tx_profile = np.zeros(self.bin_count)
                    
                    for bin_idx in range(self.bin_count):
                        tx_profile[bin_idx] = coverage_data[tx_id].get(bin_idx, 0)
                    
                    # Normalize to average=1 (if any coverage)
                    if np.sum(tx_profile) > 0:
                        tx_profile = tx_profile / np.mean(tx_profile)
                        coverage_profile += tx_profile
                
                # Average across transcripts
                coverage_profile = coverage_profile / len(tx_with_coverage)
                
                # Ensure profile has average=1
                if np.mean(coverage_profile) > 0:
                    coverage_profile = coverage_profile / np.mean(coverage_profile)
                
                # Group transcripts by length
                tx_lengths = sorted([(tx, transcript_lengths[tx]) for tx in tx_with_coverage], 
                                   key=lambda x: x[1])
                
                length_groups = {}
                group_size = max(1, len(tx_lengths) // length_bins)
                
                for i in range(length_bins):
                    start_idx = i * group_size
                    end_idx = min((i + 1) * group_size, len(tx_lengths))
                    
                    if start_idx >= len(tx_lengths):
                        break
                    
                    group_name = f"bin_{i+1}"
                    length_groups[group_name] = [tx[0] for tx in tx_lengths[start_idx:end_idx]]
                
                # Calculate length effects
                length_effect = {}
                
                for group, tx_list in length_groups.items():
                    group_counts = [transcript_counts[tx] for tx in tx_list]
                    length_effect[group] = np.mean(group_counts) if group_counts else 0
                
                # Normalize length effects to average=1
                if length_effect:
                    mean_effect = np.mean(list(length_effect.values()))
                    if mean_effect > 0:
                        length_effect = {k: v/mean_effect for k, v in length_effect.items()}
                
                # Update model parameters
                self.parameters = {
                    "profile": coverage_profile,
                    "length_effect": length_effect
                }
                
                self.logger.info("Successfully learned coverage bias model")
                return True
                
            except Exception as e:
                self.logger.error(f"Error processing BAM file: {str(e)}")
                self.logger.error(traceback.format_exc())
                return False
                
        except Exception as e:
            self.logger.error(f"Error learning coverage bias: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
        
        finally:
            # Clean up temporary directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
                self.logger.debug(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to remove temporary directory: {str(e)}")

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
        ax1.plot(x, self.parameters["profile"])
        ax1.set_xlabel("Relative position")
        ax1.set_ylabel("Relative coverage")
        ax1.set_title("Coverage bias profile")
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal line at y=1 (no bias)
        ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # Plot length effect if available
        if "length_effect" in self.parameters and self.parameters["length_effect"]:
            length_effect = self.parameters["length_effect"]
            groups = list(length_effect.keys())
            values = [length_effect[g] for g in groups]
            
            # Bar plot for length effect
            bar_positions = np.arange(len(groups))
            ax2.bar(bar_positions, values)
            ax2.set_xticks(bar_positions)
            ax2.set_xticklabels(groups, rotation=45)
            ax2.set_xlabel("Transcript length group")
            ax2.set_ylabel("Relative effect")
            ax2.set_title("Length effect on coverage")
            ax2.grid(True, alpha=0.3)
            
            # Add horizontal line at y=1 (no effect)
            ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # Add title
        fig.suptitle(f"Coverage Bias Model: {self.model_type}")
        plt.tight_layout()
        
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