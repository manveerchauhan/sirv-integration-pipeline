#!/usr/bin/env bash
# This script extracts real SIRV reads from a BAM file and prepares them for use
# with the SIRV Integration Pipeline

# Path to your aligned BAM file (replace with your actual BAM file)
BAM_FILE="/path/to/your/aligned.bam"

# Output directory
OUTPUT_DIR="./real_sirv_data"
mkdir -p "$OUTPUT_DIR"

echo "Extracting SIRV reads from BAM file..."

# Extract all reads aligned to SIRV regions
samtools view -h "$BAM_FILE" "SIRV*" > "$OUTPUT_DIR/sirv_reads.sam"

# Convert SAM to sorted BAM
samtools sort -o "$OUTPUT_DIR/sirv_reads.bam" "$OUTPUT_DIR/sirv_reads.sam"
samtools index "$OUTPUT_DIR/sirv_reads.bam"

# Convert to FASTQ format
samtools fastq "$OUTPUT_DIR/sirv_reads.bam" > "$OUTPUT_DIR/sirv_reads.fastq"

# Extract mapping information
echo "read_id,sirv_transcript,overlap_fraction,read_length,alignment_length" > "$OUTPUT_DIR/sirv_transcript_map.csv"
samtools view "$OUTPUT_DIR/sirv_reads.bam" | awk -F'\t' '{
    read_id = $1;
    ref_name = $3;
    cigar = $6;
    match_len = 0;
    
    # Parse CIGAR string to get match length
    while (cigar ~ /^[0-9]+[MIDNSHP=X]/) {
        op_len = substr(cigar, 1, match(cigar, /[^0-9]/) - 1);
        op_type = substr(cigar, match(cigar, /[^0-9]/), 1);
        cigar = substr(cigar, match(cigar, /[^0-9]/) + 1);
        
        if (op_type == "M" || op_type == "=" || op_type == "X") {
            match_len += op_len;
        }
    }
    
    # Calculate read length (sum of all operations excluding H)
    read_len = match_len;  # Simplified for this example
    
    print read_id "," ref_name ",1.0," read_len "," match_len;
}' >> "$OUTPUT_DIR/sirv_transcript_map.csv"

echo "Done! Files created in $OUTPUT_DIR:"
echo "- sirv_reads.fastq: FASTQ file with SIRV reads"
echo "- sirv_transcript_map.csv: Mapping of reads to SIRV transcripts"
echo ""
echo "You can now use these files with the SIRV Integration Pipeline:"
echo "python examples/simple_integration.py \\"
echo "    --sirv_fastq $OUTPUT_DIR/sirv_reads.fastq \\"
echo "    --sc_fastq /path/to/your/scRNA_data.fastq \\"
echo "    --sirv_map_csv $OUTPUT_DIR/sirv_transcript_map.csv \\"
echo "    --sirv_reference /path/to/sirv_reference.fa \\"
echo "    --sirv_gtf /path/to/sirv.gtf \\"
echo "    --output_dir ./output \\"
echo "    --insertion_rate 0.01"