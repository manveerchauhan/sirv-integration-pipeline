"""
Utility functions for the SIRV Integration Pipeline.

This module provides helper functions for file handling,
parameter validation, and other common tasks.
"""

import os
import sys
import logging
import subprocess
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
def setup_logger(
    name: str = "sirv_pipeline",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and/or console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        console: Whether to log to console
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log_file is provided
    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def check_external_tools() -> Dict[str, bool]:
    """
    Check if required external tools are available.
    
    Returns:
        Dict[str, bool]: Dictionary of tool names and availability
    """
    tools = {
        'minimap2': False,
        'samtools': False
    }
    
    for tool in tools:
        try:
            subprocess.run(['which', tool], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tools[tool] = True
        except subprocess.CalledProcessError:
            pass
    
    return tools


def check_file_exists(file_path: str) -> bool:
    """
    Check if a file exists and is readable.
    
    Args:
        file_path: Path to file
        
    Returns:
        bool: True if file exists and is readable
    """
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)


def check_output_writable(file_path: str) -> bool:
    """
    Check if an output file path is writable.
    
    Args:
        file_path: Path to output file
        
    Returns:
        bool: True if output path is writable
    """
    output_dir = os.path.dirname(os.path.abspath(file_path))
    
    # Check if directory exists
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            return False
    
    # Check if directory is writable
    return os.access(output_dir, os.W_OK)


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        int: File size in bytes
    """
    if check_file_exists(file_path):
        return os.path.getsize(file_path)
    else:
        return 0


def human_readable_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable size string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Human-readable size string
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.2f} {units[i]}"


def validate_insertion_rate(rate: float) -> float:
    """
    Validate insertion rate parameter.
    
    Args:
        rate: Insertion rate
        
    Returns:
        float: Validated insertion rate
        
    Raises:
        ValueError: If rate is outside valid range
    """
    if rate <= 0:
        raise ValueError("Insertion rate must be positive")
    
    if rate > 0.5:
        raise ValueError("Insertion rate cannot exceed 0.5 (50%)")
    
    return rate


def check_dependencies() -> bool:
    """
    Check if all required dependencies are available.
    
    Returns:
        bool: True if all dependencies are available
    """
    # Check external tools
    tools = check_external_tools()
    missing_tools = [tool for tool, available in tools.items() if not available]
    
    if missing_tools:
        print(f"ERROR: Missing required tools: {', '.join(missing_tools)}")
        print("Please install these tools and make sure they are in your PATH.")
        return False
    
    # Check Python dependencies
    try:
        import pandas
        import numpy
        import pysam
        from Bio import SeqIO
    except ImportError as e:
        print(f"ERROR: Missing required Python dependencies: {e}")
        print("Please install these packages using pip:")
        print("  pip install pandas numpy pysam biopython")
        return False
    
    return True