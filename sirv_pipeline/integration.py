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
import time
from Bio import SeqIO
from typing import Dict, List, Tuple, Optional, Any

# Import from other modules
from sirv_pipeline.coverage_bias import CoverageBiasModel, ReadLengthSampler, sample_read_lengths
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


def extract_cell_info(sc_fastq: str, sample_size: int = 0) -> Dict[str, int]:
    """
    Extract cell information from scRNA-seq FASTQ.
    
    This function parses cell barcodes from read headers in the FASTQ file.
    It extracts the barcode from the part before the first underscore in the read ID,
    or from the CB:Z: tag if present. It also collects statistics on UMIs if available.
    
    Args:
        sc_fastq: Path to single-cell FASTQ file
        sample_size: Number of reads to process. Set to 0 to process all reads.
        
    Returns:
        Dict[str, int]: Dictionary mapping cell barcodes to read counts
    """
    logger.info(f"Extracting cell information from {sc_fastq}")
    
    # Dictionary to store cell barcodes and their read counts
    cell_info = {}
    total_reads = 0
    processed_reads = 0
    
    # Statistics collection
    umi_stats = {}
    barcode_lengths = set()
    umi_lengths = set()
    cb_tag_present = 0
    ub_tag_present = 0
    header_formats = {
        'underscore_format': 0,
        'cb_tag_format': 0,
        'other_format': 0
    }
    
    # Determine if the file is gzipped
    opener = gzip.open if sc_fastq.endswith('.gz') else open
    
    # Process the start time to report processing duration
    start_time = time.time()
    
    with opener(sc_fastq, 'rt') as f:
        # Process the FASTQ file (4 lines per record)
        line = f.readline()
        while line and (sample_size <= 0 or processed_reads < sample_size):
            if line.startswith('@'):  # Header line
                # Parse the header line to extract the cell barcode
                header = line.strip()
                
                # Extract barcode and UMI information
                barcode = None
                umi = None
                
                # Try to extract barcode from CB:Z: tag if present
                if 'CB:Z:' in header:
                    barcode = header.split('CB:Z:')[1].split()[0].strip()
                    header_formats['cb_tag_format'] += 1
                    cb_tag_present += 1
                else:
                    # Otherwise extract from the first part of the read ID (before underscore)
                    read_id = header[1:].split()[0]  # Remove @ and get the first part
                    if '_' in read_id:
                        barcode = read_id.split('_')[0]
                        header_formats['underscore_format'] += 1
                    else:
                        barcode = read_id  # Use whole ID if no underscore
                        header_formats['other_format'] += 1
                
                # Try to extract UMI from UB:Z: tag if present
                if 'UB:Z:' in header:
                    umi = header.split('UB:Z:')[1].split()[0].strip()
                    ub_tag_present += 1
                elif '_' in read_id and len(read_id.split('_')) > 1:
                    # Try to get UMI from the second part of read ID
                    umi = read_id.split('_')[1]
                    if '#' in umi:  # Handle case where UMI is followed by #
                        umi = umi.split('#')[0]
                
                # Skip if no barcode found (shouldn't happen with our parsing)
                if not barcode:
                    processed_reads += 1
                    # Skip the next 3 lines
                    f.readline()  # sequence
                    f.readline()  # '+'
                    f.readline()  # quality
                    line = f.readline()
                    continue
                
                # Collect statistics
                barcode_lengths.add(len(barcode))
                if umi:
                    umi_lengths.add(len(umi))
                    if barcode not in umi_stats:
                        umi_stats[barcode] = set()
                    umi_stats[barcode].add(umi)
                
                # Update cell count
                if barcode not in cell_info:
                    cell_info[barcode] = 0
                cell_info[barcode] += 1
                total_reads += 1
                
                # Skip the next 3 lines (sequence, '+', quality)
                f.readline()  # sequence
                f.readline()  # '+'
                f.readline()  # quality
                
                processed_reads += 1
                
                # Log progress periodically
                if processed_reads % 100000 == 0:
                    elapsed = time.time() - start_time
                    reads_per_sec = processed_reads / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {processed_reads:,} reads, found {len(cell_info):,} cells so far ({reads_per_sec:.1f} reads/sec)")
            else:
                # Skip to the next record
                for _ in range(3):  # Skip the next 3 lines
                    f.readline()
                processed_reads += 1
            
            # Read the next header line
            line = f.readline()
    
    # Calculate summary statistics
    elapsed_time = time.time() - start_time
    reads_per_second = processed_reads / elapsed_time if elapsed_time > 0 else 0
    
    # Cell statistics
    num_cells = len(cell_info)
    mean_reads_per_cell = total_reads / num_cells if num_cells > 0 else 0
    median_reads_per_cell = np.median(list(cell_info.values())) if cell_info else 0
    max_reads_per_cell = max(cell_info.values()) if cell_info else 0
    min_reads_per_cell = min(cell_info.values()) if cell_info else 0
    
    # UMI statistics
    cells_with_umis = len(umi_stats)
    total_unique_umis = sum(len(umis) for umis in umi_stats.values())
    mean_umis_per_cell = total_unique_umis / cells_with_umis if cells_with_umis > 0 else 0
    
    # Log detailed statistics
    logger.info(f"Finished processing {processed_reads:,} reads in {elapsed_time:.2f} seconds ({reads_per_second:.1f} reads/sec)")
    logger.info(f"Found {num_cells:,} cells with {total_reads:,} total reads")
    logger.info(f"Read header formats: CB:Z tag: {header_formats['cb_tag_format']:,}, Underscore format: {header_formats['underscore_format']:,}, Other: {header_formats['other_format']:,}")
    logger.info(f"Reads per cell: Mean: {mean_reads_per_cell:.1f}, Median: {median_reads_per_cell:.1f}, Min: {min_reads_per_cell:,}, Max: {max_reads_per_cell:,}")
    
    if cells_with_umis > 0:
        logger.info(f"UMI statistics: {cells_with_umis:,} cells with UMIs, {total_unique_umis:,} total unique UMIs, {mean_umis_per_cell:.1f} mean UMIs per cell")
        if umi_lengths:
            logger.info(f"UMI lengths: {min(umi_lengths)} to {max(umi_lengths)} nucleotides")
    
    if barcode_lengths:
        logger.info(f"Barcode lengths: {min(barcode_lengths)} to {max(barcode_lengths)} nucleotides")
    
    # Generate and save histogram of reads per cell
    try:
        import matplotlib.pyplot as plt
        import os
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(sc_fastq)), "cell_stats")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot histogram of reads per cell
        plt.figure(figsize=(10, 6))
        plt.hist(list(cell_info.values()), bins=50)
        plt.xlabel("Reads per Cell")
        plt.ylabel("Number of Cells")
        plt.title(f"Distribution of Reads per Cell (n={num_cells:,})")
        
        # Add statistics to plot
        stats_text = f"Total Cells: {num_cells:,}\nTotal Reads: {total_reads:,}\n"
        stats_text += f"Mean Reads/Cell: {mean_reads_per_cell:.1f}\nMedian Reads/Cell: {median_reads_per_cell:.1f}"
        plt.figtext(0.95, 0.95, stats_text, horizontalalignment='right', verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save plot
        plot_path = os.path.join(plots_dir, "reads_per_cell.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved reads per cell histogram to {plot_path}")
    except Exception as e:
        logger.warning(f"Could not generate cell statistics plot: {str(e)}")
    
    return cell_info


def add_sirv_to_dataset(
    sc_fastq: str,
    sirv_fastq: str,
    transcript_map_file: str,
    coverage_model_file: Optional[str] = None,
    output_fastq: str = "integrated.fastq",
    tracking_file: Optional[str] = None,
    insertion_rate: float = 0.01,
    expected_file: Optional[str] = None,
    sample_size: int = 1000,
    reference_file: Optional[str] = None,
    annotation_file: Optional[str] = None,
    coverage_model: Optional[CoverageBiasModel] = None,
    coverage_model_type: str = "10x_cdna",  # New parameter 
    model_coverage_bias: bool = True,
    seed: Optional[int] = None
) -> Tuple[str, str, str]:
    """
    Add SIRV reads to an existing scRNA-seq dataset.
    
    Args:
        sc_fastq: Path to single-cell FASTQ file
        sirv_fastq: Path to SIRV FASTQ file
        transcript_map_file: Path to transcript mapping CSV file
        coverage_model_file: Path to save/load coverage model file
        output_fastq: Path to output integrated FASTQ file
        tracking_file: Path to output tracking file (default: <output_dir>/tracking.csv)
        insertion_rate: Rate of SIRV insertion (0-1, default: 0.01)
        expected_file: Path to output expected counts file (default: <output_dir>/expected_counts.csv)
        sample_size: Number of reads to sample for coverage modeling
        reference_file: Path to reference file for coverage modeling (optional)
        annotation_file: Path to annotation file for coverage learning (optional)
        coverage_model: Existing coverage model to use (optional)
        coverage_model_type: Type of coverage bias model ('10x_cdna', 'direct_rna', 'custom')
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
    
    # Step 2: Extract cell information from scRNA-seq dataset
    logger.info(f"Extracting cell information")
    cell_info = extract_cell_info(sc_fastq, sample_size=0)
    
    # Step 3: Create read length sampler and coverage bias model
    logger.info(f"Creating read length sampler and coverage bias model")
    
    # Sample read lengths from input FASTQ
    read_lengths = sample_read_lengths(sc_fastq, sample_size)
    length_sampler = ReadLengthSampler(read_lengths, seed=seed)
    
    # Initialize coverage model
    if coverage_model is None:
        if coverage_model_file is not None and os.path.exists(coverage_model_file):
            # Load existing model from file
            logger.info(f"Loading coverage model from {coverage_model_file}")
            coverage_model = CoverageBiasModel(model_type=coverage_model_type, seed=seed)
            coverage_model.load(coverage_model_file)
        elif model_coverage_bias:
            # Create a new model
            if annotation_file and reference_file:
                logger.info(f"Learning coverage model from reference data")
                # Create a BAM file from 10X Chromium data
                # This would typically be done in a separate step
                # but for now we'll use the default model
                coverage_model = CoverageBiasModel(model_type=coverage_model_type, seed=seed)
            else:
                # Use default model
                logger.info(f"Using default coverage model ({coverage_model_type})")
                coverage_model = CoverageBiasModel(model_type=coverage_model_type, seed=seed)
        else:
            # Create a uniform model (no bias)
            logger.info("Coverage bias modeling disabled, using uniform model")
            coverage_model = CoverageBiasModel(model_type="custom", seed=seed)
    
    # Save the model if a file is specified
    if coverage_model_file is not None and not os.path.exists(coverage_model_file):
        coverage_model.save(coverage_model_file)
    
    # Plot coverage bias model
    plot_file = os.path.join(output_dir, "coverage_bias.png")
    coverage_model.plot_distributions(plot_file)
    
    # Step 4: Load SIRV reads
    logger.info(f"Loading SIRV reads")
    sirv_reads = {}
    
    opener = gzip.open if sirv_fastq.endswith('.gz') else open
    with opener(sirv_fastq, 'rt') as f:
        for record in SeqIO.parse(f, 'fastq'):
            read_id = record.id
            if read_id in read_to_transcript:
                sirv_reads[read_id] = {
                    'seq': str(record.seq),
                    'qual': ''.join(chr(q+33) for q in record.letter_annotations['phred_quality']),
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
        
        # Generate a UMI for this read
        umi = barcode_gen.generate_umi()
        
        # Apply coverage bias model to determine read start/end
        original_seq = sirv_read['seq']
        original_qual = sirv_read['qual']
        original_length = len(original_seq)
        
        # Sample a read length from the distribution
        target_length = length_sampler.sample()
        
        # Apply coverage bias to get biased sequence
        biased_seq, biased_qual = coverage_model.apply_to_sequence(
            sequence=original_seq,
            quality=original_qual,
            target_length=target_length
        )
        sampled_length = len(biased_seq)
        
        # Prepend cell barcode and UMI to the sequence
        # Format: [BC][UMI][sequence]
        seq_with_tags = barcode + umi + biased_seq
        qual_with_tags = 'I' * (len(barcode) + len(umi)) + biased_qual
        
        # Create new read ID with info about original read
        new_read_id = f"SIRV_{i+1}_BC_{barcode}_UMI_{umi}_orig_{read_id}"
        
        # Write to output FASTQ
        output_handle.write(f"@{new_read_id}\n")
        output_handle.write(f"{seq_with_tags}\n")
        output_handle.write(f"+\n")
        output_handle.write(f"{qual_with_tags}\n")
        
        # Track this read for evaluation
        tracking_data.append({
            'read_id': new_read_id,
            'original_read_id': read_id,
            'barcode': barcode,
            'umi': umi,
            'sirv_transcript': sirv_read['transcript'],
            'original_length': original_length,
            'sampled_length': sampled_length
        })
        
        # Update expected counts
        count_key = (barcode, sirv_read['transcript'])
        if count_key not in expected_counts:
            expected_counts[count_key] = 0
        expected_counts[count_key] += 1
        
        # Log progress periodically
        if (i+1) % 1000 == 0:
            logger.info(f"Processed {i+1}/{n_sirv_reads} SIRV reads")
    
    # Close output file
    output_handle.close()
    
    # Step 6: Write tracking and expected count files
    logger.info(f"Writing tracking and expected count files")
    
    # Save tracking data
    tracking_df = pd.DataFrame(tracking_data)
    tracking_df.to_csv(tracking_file, index=False)
    
    # Save expected counts
    expected_counts_data = [
        {'barcode': bc, 'sirv_transcript': tx, 'expected_count': count}
        for (bc, tx), count in expected_counts.items()
    ]
    expected_df = pd.DataFrame(expected_counts_data)
    expected_df.to_csv(expected_file, index=False)
    
    logger.info(f"Integration complete. Output files:")
    logger.info(f"- Combined FASTQ: {output_fastq}")
    logger.info(f"- Tracking file: {tracking_file}")
    logger.info(f"- Expected counts: {expected_file}")
    
    return output_fastq, tracking_file, expected_file