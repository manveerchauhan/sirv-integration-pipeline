"""
SIRV Integration Pipeline

A pipeline for integrating SIRV spike-in reads into existing scRNA-seq datasets
to benchmark isoform discovery tools.
"""

# Version
__version__ = "0.1.0"

# Import main modules
from sirv_pipeline.mapping import (
    map_sirv_reads,
    create_alignment,
    process_sirv_bams,
    extract_fastq_from_bam
)

from sirv_pipeline.coverage_bias import (
    model_transcript_coverage,
    create_coverage_bias_model,
    CoverageBiasModel,
    ReadLengthSampler
)

from sirv_pipeline.integration import (
    add_sirv_to_dataset
)

from sirv_pipeline.evaluation import (
    compare_with_flames,
    generate_report
)

from sirv_pipeline.utils import (
    setup_logger,
    check_dependencies,
    validate_files,
    validate_insertion_rate
)

# Export public API
__all__ = [
    # Mapping
    'map_sirv_reads',
    'create_alignment',
    'process_sirv_bams',
    'extract_fastq_from_bam',
    
    # Coverage bias
    'model_transcript_coverage',
    'create_coverage_bias_model',
    'CoverageBiasModel',
    'ReadLengthSampler',
    
    # Integration
    'add_sirv_to_dataset',
    
    # Evaluation
    'compare_with_flames',
    'generate_report',
    
    # Utils
    'setup_logger',
    'check_dependencies',
    'validate_files',
    'validate_insertion_rate'
]