#!/usr/bin/env python3
"""
SIRV Integration Pipeline - Command Line Entry Point

This script provides a simple command-line entry point to the SIRV Integration Pipeline.
It imports and uses the core functionality from the sirv_pipeline package.
"""

import sys
import logging
from pathlib import Path
from sirv_pipeline.main import parse_args, run_pipeline, setup_logger

def main():
    """Main entry point for the SIRV Integration Pipeline."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = Path(args.output_dir) / "pipeline.log"
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(str(log_file), console_level=log_level, file_level=logging.DEBUG)
    
    logger.info("Starting SIRV Integration Pipeline")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Log all arguments
    logger.debug("Command-line arguments:")
    for arg, value in vars(args).items():
        logger.debug(f"  {arg}: {value}")
    
    # Run the pipeline
    try:
        run_pipeline(args)
        logger.info("Pipeline completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 