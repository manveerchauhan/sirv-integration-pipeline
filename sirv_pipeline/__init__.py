"""
SIRV Integration Pipeline

The SIRV Integration Pipeline is a tool for integrating SIRV reads into
single-cell RNA-seq datasets for benchmarking and analysis purposes.

Main components:
- Integration: Add SIRV reads to scRNA-seq datasets
- Coverage Bias: Model and simulate transcript coverage bias
- Evaluation: Compare with FLAMES output
"""

# Version
__version__ = "0.2.0"

# Import main functionality
from sirv_pipeline.main import run_pipeline, parse_args, setup_logger
from sirv_pipeline.utils import check_dependencies, validate_files
from sirv_pipeline.coverage_bias import CoverageBiasModel
from sirv_pipeline.integration import add_sirv_to_dataset
from sirv_pipeline.evaluation import compare_with_flames, generate_report

# Define public API
__all__ = [
    'run_pipeline',
    'parse_args',
    'setup_logger',
    'check_dependencies',
    'validate_files',
    'CoverageBiasModel',
    'add_sirv_to_dataset',
    'compare_with_flames',
    'generate_report',
]

# Package metadata
__author__ = 'Genomics Team'
__email__ = 'info@genomics.team'
__license__ = 'MIT'
__description__ = 'A pipeline for integrating SIRV reads into scRNA-seq datasets'