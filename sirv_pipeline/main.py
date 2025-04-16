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
import numpy as np

from sirv_pipeline.mapping import map_sirv_reads, create_alignment, process_sirv_bams, extract_fastq_from_bam, create_simple_gtf_from_fasta, parse_transcripts_from_gtf
from sirv_pipeline.integration import add_sirv_to_dataset
from sirv_pipeline.coverage_bias import CoverageBiasModel, model_transcript_coverage, create_coverage_bias_model
from sirv_pipeline.evaluation import compare_with_flames, generate_report
from sirv_pipeline.utils import setup_logger, check_dependencies, validate_files, create_combined_reference, fix_bam_file, analyze_bam_file, is_samtools_available


def prepare_sirv_reference(sirv_reference: str, output_dir: str) -> str:
    """
    Prepare the SIRV reference for use in the pipeline.
    
    This function copies the reference to the output directory and returns
    the path to the local copy.
    
    Args:
        sirv_reference (str): Path to SIRV reference file
        output_dir (str): Output directory
        
    Returns:
        str: Path to local reference file
    """
    logger = logging.getLogger(__name__)
    
    # Create a local copy of the reference
    local_reference = os.path.join(output_dir, "local_reference.fa")
    
    # Check if local reference already exists
    if os.path.exists(local_reference):
        logger.info(f"Local reference already exists: {local_reference}")
        return local_reference
    
    # Copy reference to output directory
    logger.info(f"Copying SIRV reference to {local_reference}")
    try:
        shutil.copy(sirv_reference, local_reference)
    except Exception as e:
        logger.error(f"Error copying SIRV reference: {str(e)}")
        raise
    
    # Ensure the reference has an index
    try:
        logger.info("Creating reference index")
        subprocess.run(["samtools", "faidx", local_reference], check=True)
    except Exception as e:
        logger.warning(f"Failed to index reference: {str(e)}")
    
    logger.info(f"SIRV reference prepared: {local_reference}")
    return local_reference


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
    """Prepare and fix the FLAMES BAM file for coverage modeling."""
    logger = logging.getLogger(__name__)
    
    if not args.learn_coverage_from or not os.path.exists(args.learn_coverage_from):
        logger.info("No FLAMES BAM file provided for coverage modeling")
        return None
    
    # Check if we've already prepared a fixed BAM
    if is_step_completed(state, "prepare_flames_bam"):
        fixed_bam_path = state["completed_steps"]["prepare_flames_bam"]["metadata"].get("fixed_bam_path")
        if fixed_bam_path and os.path.exists(fixed_bam_path):
            logger.info(f"Using previously fixed FLAMES BAM file: {fixed_bam_path}")
            return fixed_bam_path
    
    # Create output directory for fixed BAM files
    fixed_bams_dir = os.path.join(args.output_dir, "fixed_bams")
    os.makedirs(fixed_bams_dir, exist_ok=True)
    
    # Fix FLAMES BAM files
    logger.info(f"Fixing FLAMES BAM file: {args.learn_coverage_from}")
    
    # Extract base filename without extension
    bam_basename = os.path.basename(args.learn_coverage_from)
    if bam_basename.endswith(".bam"):
        bam_basename = bam_basename[:-4]
    
    # Create fixed BAM path
    fixed_bam_path = os.path.join(fixed_bams_dir, f"fixed_{bam_basename}.bam")
    
    # Run fixBAM script
    try:
        from sirv_pipeline.utils import fix_bam_file
        fixed = fix_bam_file(args.learn_coverage_from, fixed_bam_path, create_index=True)
        
        if not fixed:
            logger.error("Failed to fix FLAMES BAM file")
            return None
        
        logger.info(f"Fixed FLAMES BAM file saved to: {fixed_bam_path}")
        
        # Create index for fixed BAM
        logger.info(f"Creating index for fixed BAM file")
        subprocess.run(["samtools", "index", fixed_bam_path], check=True)
        
        # Mark step as completed
        mark_step_completed(state, "prepare_flames_bam", {
            "fixed_bam_path": fixed_bam_path
        })
        save_pipeline_state(args.output_dir, state)
        
        return fixed_bam_path
        
    except Exception as e:
        logger.error(f"Error fixing FLAMES BAM file: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def run_comparative_analysis(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    """
    Run comparative bamslam analysis to evaluate the coverage model.
    
    Args:
        args: Command-line arguments
        state: Pipeline state
    """
    logger = logging.getLogger(__name__)
    
    # Check if we've already run comparative analysis
    if is_step_completed(state, "comparative_analysis"):
        logger.info("Comparative bamslam analysis was already completed")
        return
        
    # Check if required inputs are available
    if not args.sirv_bam or not args.sc_fastq or not args.sirv_reference:
        logger.warning("Skipping comparative analysis: missing required inputs")
        return
        
    # Get paths from completed steps
    integrated_fastq = None
    tracking_file = None
    sirv_extracted_fastq = None
    
    if is_step_completed(state, "add_sirv_to_scrna"):
        integrated_fastq = state["completed_steps"]["add_sirv_to_scrna"]["metadata"].get("integrated_fastq")
        tracking_file = state["completed_steps"]["add_sirv_to_scrna"]["metadata"].get("tracking_file")
    
    if not integrated_fastq or not os.path.exists(integrated_fastq):
        logger.warning("Skipping comparative analysis: integrated FASTQ not found")
        return
    
    # Extract SIRV FASTQ file path
    if os.path.exists(os.path.join(args.output_dir, "sirv_extracted.fastq")):
        sirv_extracted_fastq = os.path.join(args.output_dir, "sirv_extracted.fastq")
    
    logger.info("Running comparative bamslam analysis...")
    
    # Create comparative bamslam directory
    comparative_dir = os.path.join(args.output_dir, "comparative_bamslam")
    os.makedirs(comparative_dir, exist_ok=True)
    
    # Set up command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "run_comparative_bamslam.py"),
        "--sirv-bam", args.sirv_bam[0] if isinstance(args.sirv_bam, list) else args.sirv_bam,
        "--integrated-fastq", integrated_fastq,
        "--sc-bam", args.learn_coverage_from if args.learn_coverage_from else args.sc_fastq,
        "--sirv-reference", args.sirv_reference,
        "--output-dir", comparative_dir
    ]
    
    # Add optional arguments if available
    if tracking_file and os.path.exists(tracking_file):
        cmd.extend(["--tracking-csv", tracking_file])
    
    if sirv_extracted_fastq and os.path.exists(sirv_extracted_fastq):
        cmd.extend(["--sirv-extracted-fastq", sirv_extracted_fastq])
    
    if args.verbose:
        cmd.append("--verbose")
    
    # Run the command
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info(f"Comparative bamslam analysis completed successfully. Results in {comparative_dir}")
        
        # Mark step as completed
        mark_step_completed(state, "comparative_analysis", {
            "output_dir": comparative_dir
        })
        save_pipeline_state(args.output_dir, state)
    except Exception as e:
        logger.error(f"Error running comparative bamslam analysis: {str(e)}")
        logger.error(traceback.format_exc())
        logger.warning("Continuing pipeline despite comparative analysis failure")


def run_integration_pipeline(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    """Run the integration mode of the SIRV pipeline."""
    logger = logging.getLogger(__name__)
    
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
        local_reference = state["completed_steps"]["prepare_sirv_reference"]["metadata"].get("local_reference")
        if local_reference and os.path.exists(local_reference):
            logger.info(f"Using existing reference: {local_reference}")
            # Make sure args.sirv_reference is also set correctly for downstream functions
            args.sirv_reference = local_reference
        else:
            local_reference = os.path.join(args.output_dir, "local_reference.fa")
            if os.path.exists(local_reference):
                logger.info(f"Using existing reference: {local_reference}")
                args.sirv_reference = local_reference
            else:
                # Need to prepare reference again
                logger.info("Preparing SIRV reference...")
                local_reference = prepare_sirv_reference(args.sirv_reference, args.output_dir)
                args.sirv_reference = local_reference
                mark_step_completed(state, "prepare_sirv_reference", {
                    "local_reference": local_reference,
                    "combined_reference": combined_reference
                })
                save_pipeline_state(args.output_dir, state)
    else:
        # Prepare SIRV reference if not already done
        logger.info("Preparing SIRV reference...")
        local_reference = prepare_sirv_reference(args.sirv_reference, args.output_dir)
        
        # If we have both SIRV and non-SIRV references, create a combined reference
        if hasattr(args, 'non_sirv_reference') and args.non_sirv_reference and hasattr(args, 'create_combined_reference') and args.create_combined_reference:
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
        if "metadata" in state["completed_steps"]["prepare_sirv_gtf"] and "gtf_path" in state["completed_steps"]["prepare_sirv_gtf"]["metadata"]:
            gtf_path = state["completed_steps"]["prepare_sirv_gtf"]["metadata"]["gtf_path"]
            if os.path.exists(gtf_path):
                logger.info(f"Using existing GTF file: {gtf_path}")
                args.sirv_gtf = gtf_path
            else:
                # GTF path in state but file doesn't exist, need to regenerate
                auto_gtf = os.path.join(args.output_dir, "local_reference.gtf")
                logger.info("GTF file not found, regenerating from FASTA reference")
                create_simple_gtf_from_fasta(local_reference, auto_gtf)
                args.sirv_gtf = auto_gtf
                mark_step_completed(state, "prepare_sirv_gtf", {"gtf_path": auto_gtf})
                save_pipeline_state(args.output_dir, state)
        else:
            # State doesn't have metadata, check for default location
            auto_gtf = os.path.join(args.output_dir, "local_reference.gtf")
            if os.path.exists(auto_gtf):
                logger.info(f"Using existing GTF file: {auto_gtf}")
                args.sirv_gtf = auto_gtf
            else:
                logger.info("GTF file not found, regenerating from FASTA reference")
                create_simple_gtf_from_fasta(local_reference, auto_gtf)
                args.sirv_gtf = auto_gtf
                mark_step_completed(state, "prepare_sirv_gtf", {"gtf_path": auto_gtf})
                save_pipeline_state(args.output_dir, state)
    elif not is_step_completed(state, "prepare_sirv_gtf"):
        # Check if GTF file is provided, if not, generate one from the FASTA
        if not args.sirv_gtf:
            auto_gtf = os.path.join(args.output_dir, "local_reference.gtf")
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
            if not hasattr(args, 'sirv_fastq') or not args.sirv_fastq:
                if not hasattr(args, 'sirv_bam') or not args.sirv_bam or len(args.sirv_bam) == 0:
                    logger.error("No SIRV input provided. Either --sirv-fastq or --sirv-bam must be specified.")
                    raise ValueError("No SIRV input provided")
            
            if hasattr(args, 'sirv_fastq') and args.sirv_fastq:
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
                raise FileNotFoundError("Required output files are missing")
            
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
            raise
    
    # Step 4: Prepare FLAMES BAM for coverage modeling
    fixed_flames_bam = None
    if hasattr(args, 'disable_coverage_bias') and not args.disable_coverage_bias and hasattr(args, 'learn_coverage_from') and args.learn_coverage_from:
        fixed_flames_bam = prepare_flames_bam(args, state)
        if fixed_flames_bam:
            logger.info(f"Using fixed FLAMES BAM for coverage modeling: {fixed_flames_bam}")
            # Update the argument to use the fixed BAM
            args.learn_coverage_from = fixed_flames_bam
    
    # Step 5: Learn coverage bias model
    coverage_model = None
    if hasattr(args, 'disable_coverage_bias') and not args.disable_coverage_bias:
        if not is_step_completed(state, "learn_coverage_bias"):
            if hasattr(args, 'learn_coverage_from') and args.learn_coverage_from and os.path.exists(args.learn_coverage_from):
                logger.info(f"Learning coverage bias from {args.learn_coverage_from}")
                
                # Initialize coverage model
                if hasattr(args, 'coverage_model') and args.coverage_model == 'custom' and args.learn_coverage_from:
                    coverage_model = CoverageBiasModel(length_bins=args.length_bins, logger=logger)
                else:
                    model_type = args.coverage_model if hasattr(args, 'coverage_model') else "10x_cdna"
                    coverage_model = CoverageBiasModel(
                        model_type=model_type,
                        bin_count=100,
                        seed=args.seed if hasattr(args, 'seed') else None
                    )
                
                # Determine which annotation file to use for coverage learning
                coverage_annotation_file = args.sirv_gtf
                if hasattr(args, 'flames_gtf') and args.flames_gtf and os.path.exists(args.flames_gtf):
                    logger.info(f"Using FLAMES GTF file for coverage modeling: {args.flames_gtf}")
                    coverage_annotation_file = args.flames_gtf
                else:
                    logger.info(f"Using SIRV GTF file for coverage modeling: {args.sirv_gtf}")
                
                # Learn from BAM file
                min_reads = args.min_reads if hasattr(args, 'min_reads') else 100
                length_bins = args.length_bins if hasattr(args, 'length_bins') else 5
                success = coverage_model.learn_from_bam(
                    bam_file=args.learn_coverage_from,
                    annotation_file=coverage_annotation_file,
                    min_reads=min_reads,
                    length_bins=length_bins
                )
                
                if not success:
                    logger.warning("Could not learn coverage bias from BAM file")
                    logger.info(f"Using default coverage model")
                    model_type = args.coverage_model if hasattr(args, 'coverage_model') else "10x_cdna"
                    coverage_model = CoverageBiasModel(model_type=model_type)
                
                # Save coverage model
                try:
                    coverage_model.save(coverage_model_file)
                    logger.info(f"Saved coverage model to {coverage_model_file}")
                except Exception as e:
                    logger.error(f"Error saving coverage model: {str(e)}")
                    logger.error(traceback.format_exc())
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
                        logger.error(traceback.format_exc())
                
                mark_step_completed(state, "learn_coverage_bias", {
                    "model_file": coverage_model_file,
                    "model_type": coverage_model.model_type
                })
                save_pipeline_state(args.output_dir, state)
            else:
                logger.info(f"Using default coverage model")
                model_type = args.coverage_model if hasattr(args, 'coverage_model') else "10x_cdna"
                coverage_model = CoverageBiasModel(model_type=model_type)
        elif is_step_completed(state, "learn_coverage_bias"):
            # Load previously learned model
            if "metadata" in state["completed_steps"]["learn_coverage_bias"]:
                model_file = state["completed_steps"]["learn_coverage_bias"]["metadata"].get("model_file")
                if model_file and os.path.exists(model_file):
                    try:
                        with open(model_file, 'rb') as f:
                            model_data = pickle.load(f)
                        
                        model_type = args.coverage_model if hasattr(args, 'coverage_model') else "10x_cdna"
                        coverage_model = CoverageBiasModel(
                            model_type=model_data.get('model_type', model_type),
                            parameters=model_data.get('parameters', {})
                        )
                        logger.info(f"Loaded coverage model from {model_file}")
                    except Exception as e:
                        logger.error(f"Error loading coverage model: {str(e)}")
                        logger.error(traceback.format_exc())
                        model_type = args.coverage_model if hasattr(args, 'coverage_model') else "10x_cdna"
                        coverage_model = CoverageBiasModel(model_type=model_type)
                else:
                    model_type = args.coverage_model if hasattr(args, 'coverage_model') else "10x_cdna"
                    coverage_model = CoverageBiasModel(model_type=model_type)
            else:
                model_type = args.coverage_model if hasattr(args, 'coverage_model') else "10x_cdna"
                coverage_model = CoverageBiasModel(model_type=model_type)
    else:
        logger.info("Coverage bias modeling disabled")
        coverage_model = None
    
    # Step 6: Generate coverage visualizations
    if hasattr(args, 'visualize_coverage') and args.visualize_coverage and not is_step_completed(state, "visualize_coverage") and coverage_model:
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
            logger.error(traceback.format_exc())
        
        mark_step_completed(state, "visualize_coverage", {"plots_dir": plots_dir})
        save_pipeline_state(args.output_dir, state)
    
    # Step 7: Add SIRV reads to scRNA-seq dataset
    if not is_step_completed(state, "add_sirv_to_scrna") and hasattr(args, 'sc_fastq') and args.sc_fastq:
        try:
            logger.info("Adding SIRV reads to scRNA-seq dataset...")
            
            # Create temporary FASTQ from BAM if needed
            sirv_fastq_for_integration = args.sirv_fastq if hasattr(args, 'sirv_fastq') else None
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
                insertion_rate=args.insertion_rate if hasattr(args, 'insertion_rate') else 0.1,
                reference_file=combined_reference or (args.non_sirv_reference if hasattr(args, 'non_sirv_reference') else None),
                annotation_file=args.sirv_gtf,
                coverage_model=coverage_model,
                coverage_model_type=args.coverage_model if hasattr(args, 'coverage_model') else "10x_cdna",
                model_coverage_bias=not args.disable_coverage_bias if hasattr(args, 'disable_coverage_bias') else True,
                seed=args.seed if hasattr(args, 'seed') else None
            )
            
            mark_step_completed(state, "add_sirv_to_scrna", {
                "integrated_fastq": integrated_fastq,
                "tracking_file": tracking_file
            })
            save_pipeline_state(args.output_dir, state)
        except Exception as e:
            logger.error(f"Error adding SIRV reads to scRNA-seq dataset: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    logger.info("SIRV Integration Pipeline completed successfully")
    return True


def run_evaluation_pipeline(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    """Run the evaluation mode of the SIRV pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Running evaluation pipeline...")
    
    # Implement evaluation mode logic here
    # This is a placeholder that can be expanded later
    
    logger.info("Evaluation completed successfully")
    return True


def run_pipeline(args: argparse.Namespace) -> bool:
    """Run the SIRV Integration Pipeline."""
    logger = logging.getLogger(__name__)
    
    # First check if config presets should be listed
    if hasattr(args, 'list_config_presets') and args.list_config_presets:
        # Implementation for listing config presets
        logger.info("Available configuration presets:")
        # List presets here
        sys.exit(0)
    
    # Initialize pipeline state
    state = load_pipeline_state(args.output_dir)
    
    # Apply configuration preset if specified
    if hasattr(args, 'config_preset') and args.config_preset:
        # Apply preset to args
        logger.info(f"Applying configuration preset: {args.config_preset}")
        # Implementation for applying preset
    
    try:
        # Step 1: Check dependencies
        logger.info("Checking dependencies...")
        check_dependencies()
        
        # Set the mode
        if hasattr(args, 'integration') and args.integration:
            logger.info("Running in integration mode")
            run_integration_pipeline(args, state)
        elif hasattr(args, 'evaluation') and args.evaluation:
            logger.info("Running in evaluation mode")
            run_evaluation_pipeline(args, state)
        else:
            logger.error("No mode specified. Use --integration or --evaluation")
            sys.exit(1)
            
        # Mark pipeline as completed
        mark_step_completed(state, "pipeline_completed")
        save_pipeline_state(args.output_dir, state)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for SIRV Integration Pipeline."""
    parser = argparse.ArgumentParser(
        description="SIRV Integration Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output directory
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory for all pipeline results"
    )
    
    # Mode selection arguments
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run in integration mode to add SIRV reads to scRNA-seq data"
    )
    
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run in diagnostics mode to analyze SIRV reads in integrated data"
    )
    
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Run in evaluation mode to assess the quality of the integration"
    )
    
    # SIRV reference arguments
    parser.add_argument(
        "--sirv-reference",
        help="SIRV reference FASTA file"
    )
    
    parser.add_argument(
        "--sirv-gtf",
        help="SIRV annotation GTF file"
    )
    
    # Input files for integration mode
    parser.add_argument(
        "--sirv-bam",
        help="BAM file(s) with SIRV reads aligned to the SIRV reference",
        nargs="+"
    )
    
    parser.add_argument(
        "--sirv-fastq",
        help="FASTQ file(s) with SIRV reads",
        nargs="+"
    )
    
    parser.add_argument(
        "--sc-fastq",
        help="FASTQ file with scRNA-seq reads"
    )
    
    # Integration options
    parser.add_argument(
        "--integration-rate",
        type=float,
        default=0.1,
        help="Target SIRV:scRNA-seq ratio for integration"
    )
    
    parser.add_argument(
        "--coverage-model",
        choices=["10x_cdna", "direct_rna", "custom", "random_forest", "ml_gradient_boosting"],
        default="random_forest",
        help="Coverage bias model to use for transcript integration. Options: "
             "'random_forest' (recommended, uses machine learning to model complex patterns), "
             "'ml_gradient_boosting' (uses gradient boosting algorithm for potentially better accuracy), "
             "'10x_cdna' (3' bias typical of 10X cDNA), "
             "'direct_rna' (5' bias typical of direct RNA), "
             "'custom' (basic model learned from input BAM file)"
    )
    
    parser.add_argument(
        "--learn-coverage-from",
        help="BAM file for learning coverage model (for custom and random_forest models)"
    )
    
    parser.add_argument(
        "--annotation-file",
        help="Annotation file (GTF/GFF) for learning coverage model"
    )
    
    parser.add_argument(
        "--min-reads-for-learning",
        type=int,
        default=100,
        help="Minimum number of reads for a transcript to be used in model learning"
    )
    
    parser.add_argument(
        "--length-bins",
        type=int,
        default=5,
        help="Number of transcript length bins for coverage model"
    )
    
    # New configuration and feature cache options
    parser.add_argument(
        "--config-file",
        help="Path to custom configuration file (JSON or YAML) for the coverage model"
    )
    
    parser.add_argument(
        "--config-preset",
        choices=["nanopore_cdna", "pacbio_isoseq", "direct_rna", "balanced"],
        help="Preset configuration for the coverage model based on sequencing technology"
    )
    
    parser.add_argument(
        "--feature-cache-file",
        help="Path to feature cache file to speed up model learning and avoid redundant calculations"
    )
    
    parser.add_argument(
        "--list-config-presets",
        action="store_true",
        help="List available configuration presets and exit"
    )
    
    # Common arguments
    common_group = parser.add_argument_group("Common Settings")
    common_group.add_argument(
        "--non-sirv-reference", type=str,
        help="Path to non-SIRV reference FASTA file (e.g., genome reference)"
    )
    common_group.add_argument(
        "--create-combined-reference", action="store_true",
        help="Create a combined SIRV and non-SIRV reference (for downstream analysis)"
    )
    common_group.add_argument(
        "--insertion-rate", type=float, default=0.1,
        help="SIRV insertion rate (0-1, default: 0.1)"
    )
    common_group.add_argument(
        "--visualize-coverage", action="store_true",
        help="Generate coverage bias visualizations"
    )
    common_group.add_argument(
        "--disable-coverage-bias", action="store_true",
        help="Disable coverage bias modeling"
    )
    common_group.add_argument(
        "--extract-features", action="store_true",
        help="Extract sequence features for ML models (for random_forest and ml_gradient_boosting models, requires BioPython and pyfaidx)"
    )
    common_group.add_argument(
        "--run-comparative-analysis", action="store_true",
        help="Run comparative bamslam analysis after integration (recommended)"
    )
    
    # Common arguments
    common_group = parser.add_argument_group("Common Settings")
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