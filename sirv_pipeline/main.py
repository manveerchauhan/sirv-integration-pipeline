"""
SIRV Integration Pipeline - main module

This is the entry point for the SIRV Integration Pipeline, which handles
command-line arguments and coordinates the pipeline's execution.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import json
import datetime
import tempfile
import traceback
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import subprocess
import shutil

from sirv_pipeline.mapping import map_sirv_reads, create_alignment, process_sirv_bams, extract_fastq_from_bam, create_simple_gtf_from_fasta, parse_transcripts_from_gtf
from sirv_pipeline.integration import add_sirv_to_dataset
from sirv_pipeline.coverage_bias import CoverageBiasModel, model_transcript_coverage
from sirv_pipeline.evaluation import compare_with_flames, generate_report
from sirv_pipeline.utils import setup_logger, check_dependencies, validate_files, create_combined_reference, fix_bam_file, analyze_bam_file


def load_pipeline_state(output_dir: str) -> Dict[str, Any]:
    """
    Load pipeline state from a state file.
    
    Args:
        output_dir (str): Path to output directory
        
    Returns:
        dict: Pipeline state
    """
    logger = logging.getLogger(__name__)
    state_file = os.path.join(output_dir, "pipeline_state.json")
    
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                logger.info(f"Loaded pipeline state from {state_file}")
                return state
        except Exception as e:
            logger.warning(f"Error loading pipeline state: {str(e)}")
    
    # Initialize with empty state
    return {
        "completed_steps": {},
        "started_at": datetime.datetime.now().isoformat(),
        "last_updated": datetime.datetime.now().isoformat()
    }


def save_pipeline_state(output_dir: str, state: Dict[str, Any]) -> None:
    """
    Save pipeline state to a state file.
    
    Args:
        output_dir (str): Path to output directory
        state (dict): Pipeline state
    """
    logger = logging.getLogger(__name__)
    state_file = os.path.join(output_dir, "pipeline_state.json")
    
    # Update timestamp
    state["last_updated"] = datetime.datetime.now().isoformat()
    
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
            logger.info(f"Saved pipeline state to {state_file}")
    except Exception as e:
        logger.warning(f"Error saving pipeline state: {str(e)}")


def mark_step_completed(state: Dict[str, Any], step_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Mark a pipeline step as completed.
    
    Args:
        state (dict): Pipeline state
        step_name (str): Name of the step
        metadata (dict, optional): Additional metadata about the step
    """
    if "completed_steps" not in state:
        state["completed_steps"] = {}
    
    state["completed_steps"][step_name] = {
        "completed_at": datetime.datetime.now().isoformat(),
        "metadata": metadata or {}
    }


def is_step_completed(state: Dict[str, Any], step_name: str) -> bool:
    """
    Check if a pipeline step has been completed.
    
    Args:
        state (dict): Pipeline state
        step_name (str): Name of the step
        
    Returns:
        bool: True if the step has been completed
    """
    return step_name in state.get("completed_steps", {})


def prepare_flames_bam(args: argparse.Namespace, state: Dict[str, Any]) -> Optional[str]:
    """
    Prepare the FLAMES BAM file for coverage bias learning.
    
    Args:
        args: Command line arguments
        state: Pipeline state
        
    Returns:
        str: Path to fixed FLAMES BAM file or None if not available
    """
    logger = logging.getLogger(__name__)
    
    # Check if this step is already completed
    if is_step_completed(state, "prepare_flames_bam"):
        fixed_bam = state["completed_steps"]["prepare_flames_bam"]["metadata"].get("fixed_bam_path")
        if fixed_bam and os.path.exists(fixed_bam) and os.path.exists(fixed_bam + ".bai"):
            logger.info(f"Using previously fixed FLAMES BAM file: {fixed_bam}")
            return fixed_bam
    
    # Check if FLAMES BAM is provided
    if not args.learn_coverage_from:
        logger.info("No FLAMES BAM file provided for coverage bias learning")
        return None
    
    flames_bam = args.learn_coverage_from
    
    # Check if the file exists
    if not os.path.exists(flames_bam):
        logger.warning(f"FLAMES BAM file not found: {flames_bam}")
        return None
    
    # Create directory for fixed files
    fixed_dir = os.path.join(args.output_dir, "fixed_bams")
    os.makedirs(fixed_dir, exist_ok=True)
    
    # Create a name for the fixed BAM file
    bam_basename = os.path.basename(flames_bam)
    fixed_bam = os.path.join(fixed_dir, f"fixed_{bam_basename}")
    
    # First, analyze the BAM file
    logger.info(f"Analyzing FLAMES BAM file: {flames_bam}")
    analysis = analyze_bam_file(flames_bam, sample_size=1000)
    
    # Log some information about the BAM file
    logger.info(f"BAM file has {analysis.get('reference_count', 0)} references")
    logger.info(f"BAM file has approximately {analysis.get('read_count', 'unknown')} reads")
    logger.info(f"BAM file is sorted: {analysis.get('is_sorted', False)}")
    logger.info(f"BAM file has index: {analysis.get('has_index', False)}")
    
    # Fix the BAM file
    logger.info(f"Fixing FLAMES BAM file: {flames_bam}")
    success = fix_bam_file(flames_bam, fixed_bam, create_index=True)
    
    if success:
        logger.info(f"Successfully fixed FLAMES BAM file: {fixed_bam}")
        mark_step_completed(state, "prepare_flames_bam", {"fixed_bam_path": fixed_bam})
        save_pipeline_state(args.output_dir, state)
        return fixed_bam
    else:
        logger.error(f"Failed to fix FLAMES BAM file: {flames_bam}")
        return None


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the SIRV Integration Pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    log_file = os.path.join(args.output_dir, "pipeline.log")
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(log_file, console_level=log_level, file_level=logging.DEBUG)
    
    # Load pipeline state for resumption
    state = load_pipeline_state(args.output_dir)
    
    # Log start information
    logger.info("Starting SIRV Integration Pipeline")
    
    # Check dependencies
    logger.info("Checking dependencies...")
    check_dependencies()
    
    # If we're resuming, show the steps that were completed
    if len(state.get("completed_steps", {})) > 0:
        logger.info("Resuming pipeline from previous run")
        for step, info in state["completed_steps"].items():
            completed_at = info.get("completed_at", "unknown time")
            logger.info(f"Step '{step}' was completed at {completed_at}")
    
    # Run the pipeline in the requested mode
    if args.integration:
        logger.info("Running in integration mode")
        
        # Define file paths for outputs
        alignment_file = os.path.join(args.output_dir, "sirv_alignment.bam")
        transcript_map_file = os.path.join(args.output_dir, "transcript_map.csv")
        integrated_fastq = os.path.join(args.output_dir, "integrated_reads.fastq")
        tracking_file = os.path.join(args.output_dir, "integration_tracking.csv")
        coverage_model_file = os.path.join(args.output_dir, "coverage_model.pkl")
        combined_reference = None
        
        # Set proper path to local_reference when resuming
        if is_step_completed(state, "prepare_sirv_reference"):
            # If resuming and reference was already prepared, set the local_reference path
            local_reference = os.path.join(args.output_dir, "local_reference.fa")
            logger.info(f"Using existing reference: {local_reference}")
            # Make sure args.sirv_reference is also set correctly for downstream functions
            args.sirv_reference = local_reference
        else:
            # Prepare SIRV reference if not already done
            if not is_step_completed(state, "prepare_sirv_reference"):
                logger.info("Preparing SIRV reference...")
                local_reference = prepare_sirv_reference(args.sirv_reference, args.output_dir)
                state["prepare_sirv_reference"] = datetime.datetime.now().isoformat()
                save_pipeline_state(args.output_dir, state)
            else:
                logger.info(f"Step 'prepare_sirv_reference' was completed at {state['prepare_sirv_reference']}")
                local_reference = os.path.join(args.output_dir, "local_reference.fa")
            
            # If we have both SIRV and non-SIRV references, create a combined reference
            if args.non_sirv_reference and args.create_combined_reference:
                combined_reference = os.path.join(args.output_dir, "combined_reference.fa")
                create_combined_reference(local_reference, args.non_sirv_reference, combined_reference)
            
            mark_step_completed(state, "prepare_sirv_reference", {
                "local_reference": local_reference,
                "combined_reference": combined_reference
            })
            save_pipeline_state(args.output_dir, state)
            
            # Update the reference to use
            args.sirv_reference = local_reference
        
        # Step 2: Prepare SIRV GTF
        if is_step_completed(state, "prepare_sirv_gtf"):
            # If resuming and GTF was already prepared, set the GTF path
            auto_gtf = os.path.join(args.output_dir, "local_reference.gtf")
            if os.path.exists(auto_gtf):
                logger.info(f"Using existing GTF file: {auto_gtf}")
                args.sirv_gtf = auto_gtf
        elif not is_step_completed(state, "prepare_sirv_gtf"):
            # Check if GTF file is provided, if not, generate one from the FASTA
            if not args.sirv_gtf:
                auto_gtf = os.path.splitext(local_reference)[0] + '.gtf'
                if not os.path.exists(auto_gtf):
                    logger.info("No GTF file provided, generating from FASTA reference")
                    create_simple_gtf_from_fasta(local_reference, auto_gtf)
                    args.sirv_gtf = auto_gtf
                else:
                    logger.info(f"Using existing GTF file: {auto_gtf}")
                    args.sirv_gtf = auto_gtf
            
            mark_step_completed(state, "prepare_sirv_gtf", {"gtf_path": args.sirv_gtf})
            save_pipeline_state(args.output_dir, state)
        
        # Step 3: Process SIRV reads from BAM or FASTQ
        if not is_step_completed(state, "process_sirv_reads"):
            try:
                # Ensure sirv_bam is not None when it's required
                if not args.sirv_fastq and (not args.sirv_bam or len(args.sirv_bam) == 0):
                    logger.error("No SIRV input provided. Either --sirv-fastq or --sirv-bam must be specified.")
                    sys.exit(1)
                
                if args.sirv_fastq:
                    # Original FASTQ workflow
                    logger.info("Mapping SIRV reads from FASTQ...")
                    map_sirv_reads(
                        args.sirv_fastq,
                        args.sirv_reference,
                        args.sirv_gtf,
                        transcript_map_file,
                        threads=args.threads
                    )
                    
                    # Create alignment
                    logger.info("Creating SIRV alignment...")
                    create_alignment(
                        args.sirv_fastq,
                        args.sirv_reference,
                        alignment_file,
                        threads=args.threads,
                        preset="map-ont"
                    )
                else:
                    # BAM input workflow
                    logger.info(f"Processing SIRV reads from {len(args.sirv_bam)} BAM file(s)...")
                    process_sirv_bams(
                        args.sirv_bam,
                        args.sirv_reference,
                        args.sirv_gtf,
                        transcript_map_file,
                        alignment_file,
                        threads=args.threads
                    )
                
                # Verify that we have created the necessary files
                if not os.path.exists(alignment_file) or not os.path.exists(transcript_map_file):
                    logger.error("Failed to process SIRV reads")
                    sys.exit(1)
                
                # Create index for alignment file if needed
                if not os.path.exists(alignment_file + ".bai"):
                    logger.info(f"Indexing BAM file: {alignment_file}")
                    subprocess.run(["samtools", "index", alignment_file], check=True)
                
                mark_step_completed(state, "process_sirv_reads", {
                    "alignment_file": alignment_file,
                    "transcript_map_file": transcript_map_file
                })
                save_pipeline_state(args.output_dir, state)
            except Exception as e:
                logger.error(f"Error processing SIRV reads: {str(e)}")
                logger.error(traceback.format_exc())
                sys.exit(1)
        
        # Step 4: Prepare FLAMES BAM for coverage modeling
        fixed_flames_bam = None
        if not args.disable_coverage_bias and args.learn_coverage_from:
            fixed_flames_bam = prepare_flames_bam(args, state)
            if fixed_flames_bam:
                logger.info(f"Using fixed FLAMES BAM for coverage modeling: {fixed_flames_bam}")
                # Update the argument to use the fixed BAM
                args.learn_coverage_from = fixed_flames_bam
        
        # Step 5: Learn coverage bias model
        coverage_model = None
        if not is_step_completed(state, "learn_coverage_bias") and not args.disable_coverage_bias:
            if args.learn_coverage_from and os.path.exists(args.learn_coverage_from):
                logger.info(f"Learning coverage bias from {args.learn_coverage_from}")
                
                # Initialize coverage model
                if args.coverage_model == 'custom' and args.learn_coverage_from:
                    coverage_model = CoverageBiasModel(length_bins=args.length_bins, logger=logger)
                else:
                    coverage_model = CoverageBiasModel(
                        model_type=args.coverage_model,
                        bin_count=100,
                        seed=args.seed
                    )
                
                # Determine which annotation file to use for coverage learning
                coverage_annotation_file = args.sirv_gtf
                if hasattr(args, 'flames_gtf') and args.flames_gtf and os.path.exists(args.flames_gtf):
                    logger.info(f"Using FLAMES GTF file for coverage modeling: {args.flames_gtf}")
                    coverage_annotation_file = args.flames_gtf
                else:
                    logger.info(f"Using SIRV GTF file for coverage modeling: {args.sirv_gtf}")
                
                # Learn from BAM file
                success = coverage_model.learn_from_bam(
                    bam_file=args.learn_coverage_from,
                    annotation_file=coverage_annotation_file,
                    min_reads=args.min_reads,
                    length_bins=args.length_bins
                )
                
                if not success:
                    logger.warning("Could not learn coverage bias from BAM file")
                    logger.info(f"Using default {args.coverage_model} coverage model")
                    coverage_model = CoverageBiasModel(model_type=args.coverage_model)
                
                # Save coverage model
                try:
                    coverage_model.save(coverage_model_file)
                    logger.info(f"Saved coverage model to {coverage_model_file}")
                except Exception as e:
                    logger.error(f"Error saving coverage model: {str(e)}")
                    # Try a simple pickle save as fallback
                    try:
                        with open(coverage_model_file, 'wb') as f:
                            pickle.dump({
                                'model_type': coverage_model.model_type,
                                'parameters': coverage_model.parameters
                            }, f)
                        logger.info(f"Saved coverage model data to {coverage_model_file}")
                    except Exception as e2:
                        logger.error(f"Failed to save coverage model: {str(e2)}")
                
                mark_step_completed(state, "learn_coverage_bias", {
                    "model_file": coverage_model_file,
                    "model_type": coverage_model.model_type
                })
                save_pipeline_state(args.output_dir, state)
            else:
                logger.info(f"Using default {args.coverage_model} coverage model")
                coverage_model = CoverageBiasModel(model_type=args.coverage_model)
        elif is_step_completed(state, "learn_coverage_bias") and not args.disable_coverage_bias:
            # Load previously learned model
            model_file = state["completed_steps"]["learn_coverage_bias"]["metadata"].get("model_file")
            if model_file and os.path.exists(model_file):
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    coverage_model = CoverageBiasModel(
                        model_type=model_data.get('model_type', args.coverage_model),
                        parameters=model_data.get('parameters', {})
                    )
                    logger.info(f"Loaded coverage model from {model_file}")
                except Exception as e:
                    logger.error(f"Error loading coverage model: {str(e)}")
                    coverage_model = CoverageBiasModel(model_type=args.coverage_model)
            else:
                coverage_model = CoverageBiasModel(model_type=args.coverage_model)
        elif args.disable_coverage_bias:
            logger.info("Coverage bias modeling disabled")
            coverage_model = None
        
        # Step 6: Generate coverage visualizations
        if args.visualize_coverage and not is_step_completed(state, "visualize_coverage") and coverage_model:
            plots_dir = os.path.join(args.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generate coverage bias plot
            coverage_plot_file = os.path.join(plots_dir, "coverage_bias.png")
            try:
                if hasattr(coverage_model, 'plot_distributions'):
                    coverage_model.plot_distributions(coverage_plot_file)
                    logger.info(f"Generated coverage bias plot: {coverage_plot_file}")
                else:
                    logger.warning("Coverage model does not have plot_distributions method")
            except Exception as e:
                logger.error(f"Error generating coverage plot: {str(e)}")
            
            mark_step_completed(state, "visualize_coverage", {"plots_dir": plots_dir})
            save_pipeline_state(args.output_dir, state)
        
        # Step 7: Add SIRV reads to scRNA-seq dataset
        if not is_step_completed(state, "add_sirv_to_scrna") and args.sc_fastq:
            try:
                logger.info("Adding SIRV reads to scRNA-seq dataset...")
                
                # Create temporary FASTQ from BAM if needed
                sirv_fastq_for_integration = args.sirv_fastq
                if not sirv_fastq_for_integration:
                    sirv_fastq_for_integration = os.path.join(args.output_dir, "sirv_extracted.fastq")
                    logger.info(f"Extracting FASTQ from BAM for integration: {sirv_fastq_for_integration}")
                    extract_fastq_from_bam(alignment_file, sirv_fastq_for_integration)
                
                # Perform integration
                add_sirv_to_dataset(
                    sc_fastq=args.sc_fastq,
                    sirv_fastq=sirv_fastq_for_integration,
                    transcript_map_file=transcript_map_file,
                    coverage_model_file=coverage_model_file,
                    output_fastq=integrated_fastq,
                    tracking_file=tracking_file,
                    insertion_rate=args.insertion_rate,
                    reference_file=combined_reference or args.non_sirv_reference,
                    annotation_file=args.sirv_gtf,
                    coverage_model=coverage_model,
                    coverage_model_type=args.coverage_model,
                    model_coverage_bias=not args.disable_coverage_bias,
                    seed=args.seed
                )
                
                mark_step_completed(state, "add_sirv_to_scrna", {
                    "integrated_fastq": integrated_fastq,
                    "tracking_file": tracking_file
                })
                save_pipeline_state(args.output_dir, state)
            except Exception as e:
                logger.error(f"Error adding SIRV reads to scRNA-seq dataset: {str(e)}")
                logger.error(traceback.format_exc())
                sys.exit(1)
        
        # Step 8: Complete pipeline
        logger.info("SIRV Integration Pipeline completed successfully")
        mark_step_completed(state, "pipeline_completed")
        save_pipeline_state(args.output_dir, state)
    
    elif args.evaluation:
        logger.info("Running in evaluation mode")
        
        # Implement evaluation mode steps here
        # ...
        
        logger.info("Evaluation completed")
    
    else:
        logger.error("No pipeline mode specified (use --integration or --evaluation)")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SIRV Integration Pipeline."""
    parser = argparse.ArgumentParser(
        description="SIRV Integration Pipeline"
    )
    
    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument(
        "--integration", action="store_true",
        help="Run in integration mode to add SIRV reads to scRNA-seq dataset"
    )
    mode_group.add_argument(
        "--evaluation", action="store_true",
        help="Run in evaluation mode to compare with FLAMES output"
    )
    
    # Integration mode arguments
    integration_group = parser.add_argument_group("Integration Mode")
    integration_group.add_argument(
        "--sirv-fastq", type=str,
        help="Path to SIRV FASTQ file"
    )
    integration_group.add_argument(
        "--sirv-bam", nargs='+',
        help="Path to SIRV BAM file(s) (can specify multiple files)"
    )
    integration_group.add_argument(
        "--sirv-reference", type=str,
        help="Path to SIRV reference FASTA file"
    )
    integration_group.add_argument(
        "--sirv-gtf", type=str,
        help="Path to SIRV GTF annotation file (optional - will be auto-generated if not provided)"
    )
    integration_group.add_argument(
        "--sc-fastq", type=str,
        help="Path to single-cell FASTQ file"
    )
    integration_group.add_argument(
        "--non-sirv-reference", type=str,
        help="Path to non-SIRV reference FASTA file (e.g., genome reference)"
    )
    integration_group.add_argument(
        "--create-combined-reference", action="store_true",
        help="Create a combined SIRV and non-SIRV reference (for downstream analysis)"
    )
    integration_group.add_argument(
        "--insertion-rate", type=float, default=0.1,
        help="SIRV insertion rate (0-1, default: 0.1)"
    )
    
    # Coverage bias modeling arguments
    coverage_group = parser.add_argument_group("Coverage Bias Modeling")
    coverage_group.add_argument(
        "--coverage-model", type=str, choices=["10x_cdna", "direct_rna", "custom"], default="10x_cdna",
        help="Type of coverage bias model to use (default: 10x_cdna)"
    )
    coverage_group.add_argument(
        "--learn-coverage-from", type=str,
        help="Learn coverage bias from BAM file"
    )
    coverage_group.add_argument(
        "--flames-gtf", type=str,
        help="FLAMES GTF annotation file to use with the FLAMES BAM for coverage modeling"
    )
    coverage_group.add_argument(
        "--visualize-coverage", action="store_true",
        help="Generate coverage bias visualizations"
    )
    coverage_group.add_argument(
        "--min-reads", type=int, default=100,
        help="Minimum reads required for bias learning (default: 100)"
    )
    coverage_group.add_argument(
        "--length-bins", type=int, default=5,
        help="Number of transcript length bins (default: 5)"
    )
    coverage_group.add_argument(
        "--disable-coverage-bias", action="store_true",
        help="Disable coverage bias modeling"
    )
    
    # Common arguments
    common_group = parser.add_argument_group("Common Settings")
    common_group.add_argument(
        "--output-dir", type=str, default="./output",
        help="Path to output directory (default: ./output)"
    )
    common_group.add_argument(
        "--threads", type=int, default=8,
        help="Number of threads for parallel processing (default: 8)"
    )
    common_group.add_argument(
        "--seed", type=int,
        help="Random seed for reproducibility"
    )
    common_group.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the SIRV Integration Pipeline."""
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()