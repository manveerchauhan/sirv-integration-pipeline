"""
SIRV Integration Pipeline for Long-Read scRNA-seq.

A pipeline for integrating SIRV spike-in reads into existing 
scRNA-seq datasets to benchmark isoform discovery tools.
"""

__version__ = "0.1.0"
__author__ = "Manveer Chauhan"
__email__ = "mschauhan@student.unimelb.edu.au"

from sirv_pipeline.mapping import map_sirv_reads, get_transcript_statistics
from sirv_pipeline.integration import add_sirv_to_dataset
from sirv_pipeline.evaluation import compare_with_flames, generate_report
from sirv_pipeline.utils import setup_logger

__all__ = [
    "map_sirv_reads",
    "get_transcript_statistics",
    "add_sirv_to_dataset",
    "compare_with_flames",
    "generate_report",
    "setup_logger"
]