"""
Utility functions for the SIRV Integration Pipeline.

This module provides helper functions for logging, validation,
and dependency checking.
"""

import os
import sys
import logging
import subprocess

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