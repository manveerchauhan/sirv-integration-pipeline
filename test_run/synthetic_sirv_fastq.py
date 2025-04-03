#!/usr/bin/env python3
"""
Script to generate synthetic SIRV and scRNA-seq test data for testing the SIRV integration pipeline.
This creates a small, manageable test dataset that mimics the structure of real data.
"""

import os
import random
import argparse
from collections import defaultdict

def generate_random_dna(length):
    """Generate a random DNA sequence of specified length."""
    return ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length))

def generate_random_quality(length, min_qual=33, max_qual=73):
    """Generate a random quality string of specified length."""
    return ''.join(chr(random.randint(min_qual, max_qual)) for _ in range(length))

def generate_sirv_fastq(output_file, num_reads=1000):
    """Generate a synthetic SIRV FASTQ file."""
    # Define SIRV transcripts with their approximate lengths
    sirv_transcripts = {
        "SIRV101": 1500, "SIRV102": 1200, "SIRV103": 1000, "SIRV201": 2000,
        "SIRV202": 1800, "SIRV301": 1700, "SIRV302": 1300, "SIRV401": 900, 
        "SIRV501": 1600, "SIRV601": 1400
    }
    
    # Relative abundances based on real SIRV mixes
    sirv_counts = {
        "SIRV101": 400, "SIRV102": 250, "SIRV103": 400, "SIRV201": 250,
        "SIRV202": 1000, "SIRV301": 1000, "SIRV302": 206, "SIRV401": 250,
        "SIRV501": 250, "SIRV601": 206
    }
    
    # Generate synthetic sequences for each transcript
    sirv_sequences = {tid: generate_random_dna(length) for tid, length in sirv_transcripts.items()}
    
    # Calculate total count for proportional sampling
    total_count = sum(sirv_counts.values())
    
    # Prepare mapping file
    mapping_file = output_file.replace('.fastq', '_transcript_map.csv')
    with open(mapping_file, 'w') as map_f:
        map_f.write("read_id,sirv_transcript,overlap_fraction,read_length,alignment_length\n")
        
        # Generate FASTQ file
        with open(output_file, 'w') as f_out:
            read_count = 0
            
            # Generate reads proportionally to counts
            for transcript, count in sirv_counts.items():
                sequence = sirv_sequences[transcript]
                reads_to_generate = round((count / total_count) * num_reads)
                
                for i in range(reads_to_generate):
                    if read_count >= num_reads:
                        break
                        
                    read_count += 1
                    read_id = f"sirv_synthetic_{transcript}_{i+1}"
                    
                    # In real ONT data, reads are often truncated or have errors
                    # Here we'll simulate some truncation
                    trunc_ratio = random.uniform(0.7, 1.0)  # Keep 70-100% of the sequence
                    trunc_len = max(100, int(len(sequence) * trunc_ratio))
                    read_seq = sequence[:trunc_len]
                    
                    # Generate quality string
                    qual = generate_random_quality(len(read_seq))
                    
                    # Write FASTQ entry
                    f_out.write(f"@{read_id}\n{read_seq}\n+\n{qual}\n")
                    
                    # Add to mapping file
                    map_f.write(f"{read_id},{transcript},1.0,{len(read_seq)},{len(read_seq)}\n")
            
            print(f"Generated {read_count} synthetic SIRV reads in {output_file}")
            print(f"Created transcript mapping in {mapping_file}")
    
    return sirv_sequences

def generate_scrna_fastq(output_file, num_reads=2000):
    """Generate a synthetic scRNA-seq FASTQ file."""
    # Define 'genes' for synthetic scRNA-seq
    genes = {
        "GENE1": 1200, "GENE2": 1500, "GENE3": 1800, "GENE4": 900, "GENE5": 2000,
        "GENE6": 1300, "GENE7": 1600, "GENE8": 1400, "GENE9": 1100, "GENE10": 1700
    }
    
    # Define cell barcodes (simulating 10x format)
    cell_barcodes = [
        "ACGTACGTACGTACGT", "TGCATGCATGCATGCA", "GTACGTACGTACGTAC",
        "CATGCATGCATGCATG", "ATGCATGCATGCATGC"
    ]
    
    # Generate sequences for each gene
    gene_sequences = {gene: generate_random_dna(length) for gene, length in genes.items()}
    
    # Generate FASTQ file
    with open(output_file, 'w') as f_out:
        # Keep track of which UMIs we've already seen for each cell+gene combination
        cell_gene_umis = defaultdict(set)
        
        # Generate reads
        for i in range(num_reads):
            # Randomly select gene and cell
            gene = random.choice(list(genes.keys()))
            barcode = random.choice(cell_barcodes)
            
            # Generate UMI (make sure it's unique for this cell+gene)
            while True:
                umi = ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(10))
                if umi not in cell_gene_umis[(barcode, gene)]:
                    cell_gene_umis[(barcode, gene)].add(umi)
                    break
            
            # Create read ID with cell barcode and UMI information
            read_id = f"scrna_{gene}_{barcode}_{umi}"
            
            # Get gene sequence and add barcode+UMI at the start
            sequence = gene_sequences[gene]
            
            # Simulate variable length reads from 3' end (like in real 10x data)
            start_pos = random.randint(0, max(0, len(sequence) - 500))
            read_seq = sequence[start_pos:]
            
            # Generate quality string
            qual = generate_random_quality(len(read_seq))
            
            # Write FASTQ entry
            f_out.write(f"@{read_id}\n{read_seq}\n+\n{qual}\n")
        
        print(f"Generated {num_reads} synthetic scRNA-seq reads in {output_file}")

def generate_sirv_reference(sirv_sequences, output_prefix):
    """Generate SIRV reference genome and GTF files."""
    # Create FASTA file
    with open(f"{output_prefix}.fa", 'w') as fasta_out:
        for transcript, sequence in sirv_sequences.items():
            fasta_out.write(f">{transcript}\n{sequence}\n")
    
    # Create GTF file
    with open(f"{output_prefix}.gtf", 'w') as gtf_out:
        for transcript, sequence in sirv_sequences.items():
            # Extract chromosome from transcript (e.g., SIRV1 from SIRV101)
            chrom = transcript[:5]
            
            # Write transcript entry
            gtf_out.write(f'{chrom}\tSIRV\ttranscript\t1\t{len(sequence)}\t.\t+\t.\ttranscript_id "{transcript}"; gene_id "{chrom}";\n')
            
            # Write exon entry (for simplicity, just one exon per transcript)
            gtf_out.write(f'{chrom}\tSIRV\texon\t1\t{len(sequence)}\t.\t+\t.\ttranscript_id "{transcript}"; gene_id "{chrom}";\n')
    
    print(f"Created SIRV reference FASTA: {output_prefix}.fa")
    print(f"Created SIRV reference GTF: {output_prefix}.gtf")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic SIRV and scRNA-seq test data')
    parser.add_argument('--sirv_reads', type=int, default=500, help='Number of SIRV reads to generate')
    parser.add_argument('--scrna_reads', type=int, default=2000, help='Number of scRNA-seq reads to generate')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate SIRV FASTQ
    sirv_fastq = os.path.join(args.output_dir, "synthetic_sirv.fastq")
    sirv_sequences = generate_sirv_fastq(sirv_fastq, args.sirv_reads)
    
    # Generate scRNA-seq FASTQ
    scrna_fastq = os.path.join(args.output_dir, "synthetic_scrna.fastq")
    generate_scrna_fastq(scrna_fastq, args.scrna_reads)
    
    # Generate SIRV reference files
    sirv_ref_prefix = os.path.join(args.output_dir, "synthetic_sirv")
    generate_sirv_reference(sirv_sequences, sirv_ref_prefix)
    
    print("\nAll synthetic test data created successfully!")
    print("\nTo run the SIRV integration pipeline with this data:")
    print(f"python /path/to/simple_integration.py \\")
    print(f"    --sirv_fastq {sirv_fastq} \\")
    print(f"    --sc_fastq {scrna_fastq} \\")
    print(f"    --sirv_reference {sirv_ref_prefix}.fa \\")
    print(f"    --sirv_gtf {sirv_ref_prefix}.gtf \\")
    print(f"    --output_dir ./output \\")
    print(f"    --insertion_rate 0.01")

if __name__ == "__main__":
    main()