#!/bin/bash
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --account="punim0646"
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mschauhan@student.unimelb.edu.au
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000
#time in days-hours:mins:sec
#SBATCH --time=0-03:00:00

module load GCCcore/11.3.0
module load minimap2/2.26
module load GCC/11.3.0
module load SAMtools/1.21

# Run minimap2 against SIRV transcriptome - change filename to correct file for each barcode
minimap2 -ax map-ont -t 8 --sam-hit-only --secondary=no /data/gpfs/projects/punim0646/sirv_genomes/sirv_transcriptome_c.fa bc1_fastq_pass | samtools view -bS - > bc01_sirv_reads.bam
samtools flagstat bc01_sirv_reads.bam

minimap2 -ax map-ont -t 8 --sam-hit-only --secondary=no /data/gpfs/projects/punim0646/sirv_genomes/sirv_transcriptome_c.fa bc2_fastq_pass | samtools view -bS - > bc02_sirv_reads.bam
samtools flagstat bc02_sirv_reads.bam

minimap2 -ax map-ont -t 8 --sam-hit-only --secondary=no /data/gpfs/projects/punim0646/sirv_genomes/sirv_transcriptome_c.fa bc3_fastq_pass | samtools view -bS - > bc03_sirv_reads.bam
samtools flagstat bc03_sirv_reads.bam