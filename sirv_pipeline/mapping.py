"""
SIRV read mapping module for the SIRV Integration Pipeline.

This module handles mapping SIRV reads to reference genomes and 
identifying transcript of origin for each read.
"""

import os
import logging
import tempfile
import subprocess
import pandas as pd
import pysam
from typing import Dict, List, Union, Optional
import shutil

from sirv_pipeline.utils import validate_files

# Set up logger
logger = logging.getLogger(__name__)

# Define full paths to executables
MINIMAP2_PATH = "/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/minimap2/2.26/bin/minimap2"
SAMTOOLS_PATH = "/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/SAMtools/1.21/bin/samtools"

def create_alignment(
    fastq_file: str,
    reference_file: str,
    output_bam: str,
    threads: int = 8,
    preset: str = "map-ont"
) -> bool:
    """
    Create alignment file for a FASTQ against a reference.
    
    Args:
        fastq_file: Path to FASTQ file
        reference_file: Path to reference file
        output_bam: Path to output BAM file
        threads: Number of threads to use
        preset: Minimap2 preset (default: map-ont)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate input files
        validate_files(fastq_file, reference_file, mode='r')
        validate_files(output_bam, mode='w')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_bam)), exist_ok=True)
        
        logger.info(f"Running alignment with minimap2 using preset: {preset}")
        
        # Run minimap2 and pipe directly to samtools to create BAM file
        cmd = [
            MINIMAP2_PATH, "-ax", preset, 
            "-t", str(threads),
            "--sam-hit-only",
            "--secondary=no", 
            reference_file, fastq_file
        ]
        
        samtools_cmd = [
            SAMTOOLS_PATH, "view", "-bS", "-", "-o", output_bam
        ]
        
        # Run pipeline: minimap2 | samtools view
        minimap_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        samtools_process = subprocess.Popen(samtools_cmd, stdin=minimap_process.stdout)
        
        # Close the stdout pipe of minimap_process and get the return code
        minimap_process.stdout.close()
        minimap_return_code = minimap_process.wait()
        samtools_return_code = samtools_process.wait()
        
        if minimap_return_code != 0:
            logger.error(f"Minimap2 failed with return code {minimap_return_code}")
            return False
            
        if samtools_return_code != 0:
            logger.error(f"Samtools view failed with return code {samtools_return_code}")
            return False
        
        # Sort the BAM file before indexing
        sorted_bam = output_bam + ".sorted.bam"
        logger.info("Sorting BAM file")
        sort_cmd = [SAMTOOLS_PATH, "sort", "-o", sorted_bam, output_bam]
        sort_result = subprocess.run(sort_cmd, check=True)
        
        if sort_result.returncode != 0:
            logger.error(f"Samtools sort failed with return code {sort_result.returncode}")
            return False
            
        # Replace the original BAM with the sorted one
        os.replace(sorted_bam, output_bam)
        
        # Index the BAM file
        logger.info("Indexing BAM file")
        subprocess.run([SAMTOOLS_PATH, "index", output_bam], check=True)
        
        # Generate statistics
        logger.info("Generating alignment statistics")
        result = subprocess.run([SAMTOOLS_PATH, "flagstat", output_bam], 
                               check=True, capture_output=True, text=True)
        stats = result.stdout
        
        # Log statistics
        for line in stats.splitlines():
            logger.info(f"Alignment stats: {line}")
        
        logger.info(f"Alignment complete: {output_bam}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating alignment: {e}")
        return False


def map_sirv_reads(
    sirv_fastq: str,
    sirv_reference: str,
    sirv_gtf: str,
    output_csv: str,
    threads: int = 8,
    min_overlap: float = 0.5,
    keep_temp: bool = False
) -> str:
    """
    Map SIRV reads to reference and identify transcript of origin.
    
    Args:
        sirv_fastq: Path to SIRV FASTQ file
        sirv_reference: Path to SIRV reference genome
        sirv_gtf: Path to SIRV annotation GTF
        output_csv: Path to output CSV file for transcript mappings
        threads: Number of threads for parallel processing
        min_overlap: Minimum overlap fraction required for transcript assignment
        keep_temp: Keep temporary files for debugging
        
    Returns:
        str: Path to the output CSV file with read-to-transcript mappings
    """
    # Validate input files
    validate_files(sirv_fastq, sirv_reference, sirv_gtf, mode='r')
    validate_files(output_csv, mode='w')
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Creating temporary directory: {temp_dir}")

    try:
        # Define output files
        prefix = os.path.basename(sirv_fastq).split('.')[0]
        bam_file = os.path.join(temp_dir, f"{prefix}.bam")
        
        # Align reads with the exact parameters from the SLURM script
        logger.info(f"Aligning SIRV reads using minimap2 with map-ont preset and {threads} threads")
        if not create_alignment(
            sirv_fastq, 
            sirv_reference, 
            bam_file, 
            threads=threads,
            preset="map-ont"
        ):
            raise RuntimeError("Failed to align SIRV reads")
        
        # Load transcript info from GTF
        transcripts = parse_transcripts_from_gtf(sirv_gtf)
        logger.info(f"Loaded {len(transcripts)} transcripts from GTF")
        
        # Process BAM to identify transcripts
        mappings = assign_transcripts_to_reads(bam_file, transcripts, min_overlap)
        
        # Save mappings to CSV
        df = pd.DataFrame(mappings)
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        logger.info(f"Found {len(df)} SIRV reads with transcript assignments")
        if not df.empty:
            logger.info(f"Unique SIRV transcripts identified: {df['sirv_transcript'].nunique()}")
        else:
            logger.warning("No SIRV transcripts were identified. Check your input files and alignment parameters.")
        
        return output_csv
        
    finally:
        # Clean up temporary files unless keep_temp is True
        if not keep_temp:
            logger.debug(f"Removing temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        else:
            logger.info(f"Keeping temporary files in: {temp_dir}")


def parse_transcripts_from_gtf(gtf_file: str) -> Dict[str, Dict]:
    """
    Parse transcript information from GTF file.
    
    Args:
        gtf_file: Path to GTF annotation file
        
    Returns:
        Dict[str, Dict]: Dictionary mapping transcript IDs to transcript information
    """
    transcripts = {}
    
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'transcript':
                continue
                
            # Extract transcript_id
            attr_str = fields[8]
            tx_id = None
            
            for attr in attr_str.split(';'):
                attr = attr.strip()
                if attr.startswith('transcript_id'):
                    # Handle both quote formats
                    if '"' in attr:
                        tx_id = attr.split('"')[1]
                    else:
                        tx_id = attr.split(' ')[1]
                    break
            
            if tx_id:
                transcripts[tx_id] = {
                    'chrom': fields[0],
                    'start': int(fields[3]),
                    'end': int(fields[4]),
                    'strand': fields[6]
                }
    
    return transcripts


def assign_transcripts_to_reads(
    bam_file: str, 
    transcripts: Dict[str, Dict], 
    min_overlap: float = 0.5
) -> List[Dict]:
    """
    Process BAM file to assign reads to transcripts.
    
    Args:
        bam_file: Path to BAM file
        transcripts: Dictionary of transcript information
        min_overlap: Minimum overlap fraction required for assignment
    Returns:
        List[Dict]: List of dictionaries with read to transcript mappings
    """
    mappings = []

    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for read in bam:
            if read.is_unmapped:
                continue

            chrom = read.reference_name
            start = read.reference_start
            end = read.reference_end or (read.reference_start + read.query_length)
            
            # Find best matching transcript 
            best_tx = None
            best_overlap = 0
            
            for tx_id, tx in transcripts.items():
                if tx['chrom'] == chrom:
                    # Calculate overlap
                    overlap_start = max(start, tx['start'])
                    overlap_end = min(end, tx['end'])
                    overlap = max(0, overlap_end - overlap_start)
                    
                    # Calculate overlap fraction relative to read length
                    read_len = end - start
                    if read_len > 0:
                        overlap_fraction = overlap / read_len
                        if overlap_fraction > best_overlap:
                            best_overlap = overlap_fraction
                            best_tx = tx_id
            
            # Only save mappings that meet the minimum overlap threshold
            if best_tx and best_overlap >= min_overlap:
                mappings.append({
                    'read_id': read.query_name,
                    'sirv_transcript': best_tx,
                    'overlap_fraction': best_overlap,
                    'read_length': read.query_length,
                    'alignment_length': end - start
                })

    return mappings


def get_transcript_statistics(mapping_csv: str) -> Dict[str, Union[int, Dict]]:
    """
    Get statistics about transcript assignments.
    
    Args:
        mapping_csv: Path to mapping CSV file
        
    Returns:
        Dict: Dictionary with transcript statistics
    """
    validate_files(mapping_csv, mode='r')
    
    df = pd.read_csv(mapping_csv)
    
    stats = {'total_reads': len(df)}
    
    if not df.empty:
        stats.update({
            'unique_transcripts': df['sirv_transcript'].nunique(),
            'reads_per_transcript': df.groupby('sirv_transcript').size().to_dict()
        })
    else:
        stats.update({
            'unique_transcripts': 0,
            'reads_per_transcript': {}
        })
    
    return stats


def extract_fastq_from_bam(bam_file: str, output_fastq: str) -> str:
    """
    Extract FASTQ reads from a BAM file using samtools.
    
    Args:
        bam_file: Path to BAM file
        output_fastq: Path to output FASTQ file
        
    Returns:
        str: Path to the output FASTQ file
    """
    try:
        # Validate input files
        validate_files(bam_file, mode='r')
        validate_files(output_fastq, mode='w')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_fastq)), exist_ok=True)
        
        logger.info(f"Extracting FASTQ from BAM: {bam_file}")
        
        # Run samtools fastq to extract reads
        cmd = [SAMTOOLS_PATH, "fastq", bam_file, "-o", output_fastq]
        
        subprocess.run(cmd, check=True)
        
        # Check if the output file was created
        if not os.path.exists(output_fastq):
            raise FileNotFoundError(f"Failed to create FASTQ file: {output_fastq}")
        
        # Count the number of reads extracted
        read_count = 0
        with open(output_fastq, 'r') as f:
            for line in f:
                if line.startswith('@'):
                    read_count += 1
        
        logger.info(f"Extracted {read_count} reads to FASTQ: {output_fastq}")
        return output_fastq
        
    except Exception as e:
        logger.error(f"Error extracting FASTQ from BAM: {e}")
        raise


def process_sirv_bams(
    bam_files: List[str],
    sirv_reference: str,
    sirv_gtf: str,
    output_csv: str,
    merged_bam: str,
    threads: int = 8,
    min_overlap: float = 0.5
) -> str:
    """
    Process one or more SIRV BAM files to identify transcript of origin.
    If multiple BAM files are provided, they will be merged.
    
    Args:
        bam_files: List of paths to SIRV BAM files
        sirv_reference: Path to SIRV reference genome
        sirv_gtf: Path to SIRV annotation GTF
        output_csv: Path to output CSV file for transcript mappings
        merged_bam: Path to output merged BAM file
        threads: Number of threads for parallel processing
        min_overlap: Minimum overlap fraction required for transcript assignment
        
    Returns:
        str: Path to the output CSV file with read-to-transcript mappings
    """
    # Validate input files
    for bam_file in bam_files:
        validate_files(bam_file, mode='r')
    validate_files(sirv_reference, sirv_gtf, mode='r')
    validate_files(output_csv, merged_bam, mode='w')
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Creating temporary directory: {temp_dir}")

    try:
        # If there's only one BAM file, just use it directly
        if len(bam_files) == 1:
            logger.info(f"Using single BAM file: {bam_files[0]}")
            
            # Create a symlink or copy the BAM file to the merged_bam path
            if os.path.abspath(bam_files[0]) != os.path.abspath(merged_bam):
                logger.info(f"Copying BAM file to: {merged_bam}")
                shutil.copy2(bam_files[0], merged_bam)
                
                # Check if the BAM file has an index
                bam_index = bam_files[0] + ".bai"
                merged_bam_index = merged_bam + ".bai"
                if os.path.exists(bam_index):
                    logger.info(f"Copying BAM index to: {merged_bam_index}")
                    shutil.copy2(bam_index, merged_bam_index)
                else:
                    # Sort the BAM file before indexing if no index exists
                    sorted_bam = merged_bam + ".sorted.bam"
                    logger.info(f"Sorting BAM file: {merged_bam}")
                    sort_cmd = [SAMTOOLS_PATH, "sort", "-o", sorted_bam, merged_bam]
                    subprocess.run(sort_cmd, check=True)
                    
                    # Replace the original BAM with the sorted one
                    os.replace(sorted_bam, merged_bam)
                    
                    # Index the BAM file
                    logger.info(f"Indexing BAM file: {merged_bam}")
                    subprocess.run([SAMTOOLS_PATH, "index", merged_bam], check=True)
            
        else:
            # We need to merge multiple BAM files
            logger.info(f"Merging {len(bam_files)} BAM files")
            
            # Merge BAM files using samtools
            merge_cmd = [SAMTOOLS_PATH, "merge", "-f", "-@", str(threads), merged_bam] + bam_files
            subprocess.run(merge_cmd, check=True)
            
            # Sort the BAM file before indexing
            sorted_bam = merged_bam + ".sorted.bam"
            logger.info(f"Sorting BAM file: {merged_bam}")
            sort_cmd = [SAMTOOLS_PATH, "sort", "-o", sorted_bam, merged_bam]
            subprocess.run(sort_cmd, check=True)
            
            # Replace the original BAM with the sorted one
            os.replace(sorted_bam, merged_bam)
            
            logger.info(f"Indexing BAM file: {merged_bam}")
            subprocess.run([SAMTOOLS_PATH, "index", merged_bam], check=True)
        
        # Load transcript info from GTF
        transcripts = parse_transcripts_from_gtf(sirv_gtf)
        logger.info(f"Loaded {len(transcripts)} transcripts from GTF")
        
        # Process BAM to identify transcripts
        mappings = assign_transcripts_to_reads(merged_bam, transcripts, min_overlap)
        
        # Save mappings to CSV
        df = pd.DataFrame(mappings)
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        logger.info(f"Found {len(df)} SIRV reads with transcript assignments")
        if not df.empty:
            logger.info(f"Unique SIRV transcripts identified: {df['sirv_transcript'].nunique()}")
        else:
            logger.warning("No SIRV transcripts were identified. Check your input files and alignment parameters.")
        
        return output_csv
        
    except Exception as e:
        logger.error(f"Error processing SIRV BAMs: {e}")
        raise
    finally:
        # Clean up temporary directory
        logger.debug(f"Removing temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

# Modification to create_simple_gtf_from_fasta function
def create_simple_gtf_from_fasta(fasta_file, output_gtf):
    """Create a simple GTF file from a FASTA reference file."""
    import pysam
    import os
    import subprocess
    import shutil
    
    logger.info(f"Creating simple GTF file from FASTA: {fasta_file}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output_gtf))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a local copy of the FASTA in the output directory
    local_fasta = os.path.join(output_dir, "local_reference.fa")
    shutil.copy2(fasta_file, local_fasta)
    
    # Index the local copy
    logger.info(f"Indexing local copy of FASTA file: {local_fasta}")
    pysam.faidx(local_fasta)
    
    # Create GTF file
    with open(output_gtf, 'w') as gtf:
        # Write header
        gtf.write('##gff-version 3\n')
        
        # Read FASTA index
        with open(f"{local_fasta}.fai", 'r') as fai:
            for line in fai:
                fields = line.strip().split('\t')
                seq_id = fields[0]
                seq_length = fields[1]
                
                # Write transcript feature
                gtf.write(f"{seq_id}\tSIRV\ttranscript\t1\t{seq_length}\t.\t+\t.\ttranscript_id \"{seq_id}\"; gene_id \"{seq_id}\";\n")
                
                # Write exon feature
                gtf.write(f"{seq_id}\tSIRV\texon\t1\t{seq_length}\t.\t+\t.\ttranscript_id \"{seq_id}\"; gene_id \"{seq_id}\";\n")
    
    logger.info(f"Created GTF file: {output_gtf}")
    return output_gtf