"""
Main entry point for the SIRV Integration Pipeline.

This module allows the package to be executed with python -m sirv_pipeline
or when installed as a command-line tool.
"""

import sys
from sirv_pipeline.main import main as main_func
from sirv_pipeline import __version__

# Expose the main function for entry point
def main():
    """Entry point for the command-line tool."""
    # Handle version flag directly for quick access
    if len(sys.argv) == 2 and sys.argv[1] == "--version":
        print(f"SIRV Integration Pipeline version {__version__}")
        return
    
    # Otherwise, run the main function
    main_func()

if __name__ == "__main__":
    main() 