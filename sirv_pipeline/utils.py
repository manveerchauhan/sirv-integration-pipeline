"""
Utility functions for the SIRV Integration Pipeline.

This module provides helper functions for logging, validation,
and dependency checking.
"""

import os
import sys
import logging
import subprocess
import tempfile
import shutil
from typing import List, Dict, Optional, Union, Tuple, Any

def setup_logger(
    log_file: str = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    name: str = "sirv_pipeline"
) -> logging.Logger:
    """
    Set up a logger with file and/or console handlers.
    
    Args:
        log_file: Path to log file (optional)
        console_level: Logging level for console output
        file_level: Logging level for file output
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))  # Set to the more detailed level
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def validate_insertion_rate(rate: float) -> float:
    """Validate insertion rate parameter."""
    if rate <= 0:
        raise ValueError("Insertion rate must be positive")
    
    if rate > 0.5:
        raise ValueError("Insertion rate cannot exceed 0.5 (50%)")
    
    return rate


def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    # Define full paths to executables
    tool_paths = {
        'minimap2': "/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/minimap2/2.26/bin/minimap2",
        'samtools': "/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/SAMtools/1.21/bin/samtools"
    }
    
    # Check external tools
    missing_tools = []
    for tool, path in tool_paths.items():
        if not os.path.exists(path) or not os.access(path, os.X_OK):
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"ERROR: Missing required tools: {', '.join(missing_tools)}")
        print("Please load the appropriate modules or install these tools.")
        return False
    
    # Check Python dependencies
    missing_packages = []
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'pysam': 'pysam',
        'Bio.SeqIO': 'biopython',
        'matplotlib.pyplot': 'matplotlib'
    }
    
    for module, package in required_packages.items():
        try:
            __import__(module.split('.')[0])
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"ERROR: Missing required Python dependencies: {', '.join(missing_packages)}")
        print("Please install these packages using pip:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    return True


def is_samtools_available() -> bool:
    """
    Check if samtools is available in the system.
    
    Returns:
        bool: True if samtools is available, False otherwise
    """
    try:
        # Try to run a simple samtools command
        result = subprocess.run(
            ["samtools", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def validate_files(*files, mode='r') -> bool:
    """
    Validate that files exist and have appropriate permissions.
    
    Args:
        *files: List of file paths to validate
        mode: 'r' for read access, 'w' for write access
        
    Returns:
        bool: True if all files are valid
        
    Raises:
        FileNotFoundError: If a file does not exist (for read mode)
        PermissionError: If a file cannot be accessed with the requested mode
    """
    for file_path in files:
        if mode == 'r':
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"File is not readable: {file_path}")
        elif mode == 'w':
            # For write mode, check if directory is writable
            dir_path = os.path.dirname(os.path.abspath(file_path))
            os.makedirs(dir_path, exist_ok=True)
            if not os.access(dir_path, os.W_OK):
                raise PermissionError(f"Directory is not writable: {dir_path}")
    
    return True


def create_combined_reference(sirv_reference: str, non_sirv_reference: str, output_file: str) -> str:
    """
    Combine SIRV and non-SIRV reference files into a single FASTA.
    
    Args:
        sirv_reference: Path to SIRV reference FASTA file
        non_sirv_reference: Path to non-SIRV reference FASTA file
        output_file: Path to output combined reference file
        
    Returns:
        str: Path to combined reference file
    """
    logger.info(f"Creating combined reference from {sirv_reference} and {non_sirv_reference}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Open output file
    with open(output_file, 'w') as out_f:
        # First add SIRV reference
        with open(sirv_reference, 'r') as sirv_f:
            for line in sirv_f:
                out_f.write(line)
        
        # Add a newline between files if needed
        out_f.write('\n')
        
        # Then add non-SIRV reference
        with open(non_sirv_reference, 'r') as non_sirv_f:
            for line in non_sirv_f:
                out_f.write(line)
    
    logger.info(f"Combined reference created: {output_file}")
    return output_file


def fix_bam_file(input_bam: str, output_bam: str, create_index: bool = True) -> bool:
    """
    Fix common issues with BAM files, such as unsorted reads, missing index, etc.
    
    Args:
        input_bam (str): Path to input BAM file
        output_bam (str): Path to output fixed BAM file
        create_index (bool): Whether to create an index for the fixed BAM file
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Fixing BAM file: {input_bam} -> {output_bam}")
    
    # Check if the output already exists
    if os.path.exists(output_bam) and (not create_index or os.path.exists(output_bam + '.bai')):
        logger.info(f"Using existing fixed BAM file: {output_bam}")
        return True
        
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_bam)), exist_ok=True)
    
    try:
        # Create a temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="bam_fix_")
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Step 1: Filter out problematic reads
        filtered_bam = os.path.join(temp_dir, "filtered.bam")
        logger.info(f"Filtering out problematic reads: {filtered_bam}")
        
        try:
            # Use samtools to filter out secondary/supplementary alignments
            filter_cmd = ["samtools", "view", "-h", "-b", "-F", "0x900", input_bam, "-o", filtered_bam]
            subprocess.run(filter_cmd, check=True)
            logger.info(f"Successfully filtered BAM file: {filtered_bam}")
        except Exception as e:
            logger.error(f"Error filtering BAM file: {str(e)}")
            logger.warning(f"Will try to proceed with original BAM file")
            filtered_bam = input_bam
        
        # Step 2: Sort the BAM file
        sorted_bam = os.path.join(temp_dir, "sorted.bam")
        logger.info(f"Sorting BAM file: {filtered_bam} -> {sorted_bam}")
        
        try:
            sort_cmd = ["samtools", "sort", "-o", sorted_bam, filtered_bam]
            subprocess.run(sort_cmd, check=True)
            logger.info(f"Successfully sorted BAM file: {sorted_bam}")
        except Exception as e:
            logger.error(f"Error sorting BAM file: {str(e)}")
            return False
        
        # Step 3: Copy to final destination
        shutil.copyfile(sorted_bam, output_bam)
        logger.info(f"Copied sorted BAM file to destination: {output_bam}")
        
        # Step 4: Create index if requested
        if create_index:
            logger.info(f"Creating index for BAM file: {output_bam}")
            try:
                index_cmd = ["samtools", "index", output_bam]
                subprocess.run(index_cmd, check=True)
                logger.info(f"Successfully created index: {output_bam}.bai")
            except Exception as e:
                logger.warning(f"Error creating BAM index: {str(e)}")
                logger.warning("Will continue without index")
        
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error removing temporary directory: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error fixing BAM file: {str(e)}")
        return False


def analyze_bam_file(bam_file: str, sample_size: int = 1000) -> Dict[str, Any]:
    """
    Analyze BAM file structure and content.
    
    Args:
        bam_file (str): Path to BAM file
        sample_size (int): Number of reads to sample for detailed analysis
        
    Returns:
        dict: Analysis results
    """
    import pysam
    from collections import Counter
    
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing BAM file: {bam_file}")
    
    results = {
        "file_path": bam_file,
        "file_exists": os.path.exists(bam_file),
        "file_size": os.path.getsize(bam_file) if os.path.exists(bam_file) else 0,
        "is_sorted": False,
        "has_index": False,
        "reference_count": 0,
        "references": [],
        "read_count": 0,
        "flags": {},
        "references_distribution": {},
        "unmapped_reads": 0,
        "secondary_alignments": 0,
        "supplementary_alignments": 0
    }
    
    if not results["file_exists"]:
        logger.error(f"BAM file not found: {bam_file}")
        return results
        
    try:
        # Check if index exists
        results["has_index"] = os.path.exists(bam_file + ".bai")
        
        # Open BAM file
        bam = pysam.AlignmentFile(bam_file, "rb")
        
        # Get header info
        header = bam.header
        
        # Check if sorted
        if hasattr(header, 'get') and header.get('HD'):
            so = header.get('HD', {}).get('SO')
            results["is_sorted"] = so == "coordinate"
        
        # Get reference information
        results["reference_count"] = len(bam.references) if bam.references else 0
        results["references"] = list(bam.references)[:10] if bam.references else []
        
        # Count total reads
        try:
            results["read_count"] = bam.count()
        except Exception as e:
            logger.warning(f"Could not count total reads: {str(e)}")
            results["read_count"] = None
            
        # Sample reads for analysis
        flags_counter = Counter()
        refs_counter = Counter()
        unmapped = 0
        secondary = 0
        supplementary = 0
        
        # Process a sample of reads
        i = 0
        for read in bam.fetch(until_eof=True):
            if i >= sample_size:
                break
                
            i += 1
            flags_counter[read.flag] += 1
            
            if read.is_unmapped:
                unmapped += 1
            if read.is_secondary:
                secondary += 1
            if read.is_supplementary:
                supplementary += 1
                
            if read.reference_name:
                refs_counter[read.reference_name] += 1
        
        results["flags"] = {str(f): c for f, c in flags_counter.most_common(10)}
        results["references_distribution"] = {str(r): c for r, c in refs_counter.most_common(10)}
        results["unmapped_reads"] = unmapped
        results["secondary_alignments"] = secondary
        results["supplementary_alignments"] = supplementary
        
        bam.close()
        logger.info(f"Successfully analyzed BAM file: {bam_file}")
        
    except Exception as e:
        logger.error(f"Error analyzing BAM file: {str(e)}")
        
    return results