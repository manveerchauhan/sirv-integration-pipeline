"""
SIRV Integration Pipeline - Main Entry Point

This module serves as the entry point when the package is run as a module.
Example: python -m sirv_pipeline [arguments]
"""

import sys
import logging
from sirv_pipeline.main import parse_args, run_pipeline, setup_logger

def main():
    """Main entry point for the SIRV Integration Pipeline when run as a module."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    log_file = args.output_dir + "/pipeline.log" if args.output_dir else None
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(log_file, console_level=log_level, file_level=logging.DEBUG)
    
    # Run the pipeline
    try:
        run_pipeline(args)
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 