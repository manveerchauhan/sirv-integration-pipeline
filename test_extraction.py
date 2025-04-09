#!/usr/bin/env python3

from sirv_pipeline.mapping import extract_fastq_from_bam
import os
import sys

if len(sys.argv) != 3:
    print("Usage: python test_extraction.py <input_bam> <output_fastq>")
    sys.exit(1)

bam_file = sys.argv[1]
output_fastq = sys.argv[2]

print(f"Extracting reads from {bam_file} to {output_fastq}...")
try:
    extract_fastq_from_bam(bam_file, output_fastq)
    read_count = 0
    with open(output_fastq, 'r') as f:
        for line in f:
            if line.startswith('@'):
                read_count += 1
    print(f"Success! Extracted {read_count} reads.")
except Exception as e:
    print(f"Error: {e}")
