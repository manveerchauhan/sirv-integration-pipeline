def learn_from_bam(self, bam_file, annotation_file, 
                   min_reads=100, length_bins=5,
                   transcript_subset=None):
    """
    Learn coverage bias model from aligned BAM file.
    
    Args:
        bam_file: Path to BAM file with aligned reads
        annotation_file: Path to GTF/GFF annotation file
        min_reads: Minimum number of reads for a transcript to be included
        length_bins: Number of transcript length bins to create
        transcript_subset: Optional list of transcript IDs to focus on
        
    Returns:
        self: For method chaining
    """
    logger.info(f"Learning coverage bias from BAM file: {bam_file}")
    
    # Ensure BAM file is properly prepared for processing
    bam_file = self._prepare_bam_file(bam_file)
    if not bam_file:
        logger.error("Failed to prepare BAM file. Using default coverage model.")
        if self.model_type == "10x_cdna":
            self._init_default_10x_cdna_model()
        else:
            self._init_default_direct_rna_model()
        return self
    
    # 1. Parse annotation file to get transcript information
    transcripts = self._parse_annotation(annotation_file)
    logger.info(f"Loaded {len(transcripts)} transcripts from annotation")
    
    # 2. Process BAM file to extract read positions
    positions = []
    transcript_reads = {}
    total_reads = 0
    skipped_reads = 0
    
    try:
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for read in bam:
                total_reads += 1
                
                # Skip unmapped, supplementary, and secondary alignments
                if read.is_unmapped or read.is_supplementary or read.is_secondary:
                    skipped_reads += 1
                    continue
                    
                transcript_id = read.reference_name
                
                # Skip if not in our subset of interest
                if transcript_subset and transcript_id not in transcript_subset:
                    skipped_reads += 1
                    continue
                    
                # Skip if transcript not in annotation
                if transcript_id not in transcripts:
                    skipped_reads += 1
                    continue
                    
                transcript_length = transcripts[transcript_id]['length']
                
                # Skip very short transcripts
                if transcript_length < 200:
                    skipped_reads += 1
                    continue
                    
                # Calculate normalized positions (0-1 range)
                try:
                    start_norm = read.reference_start / transcript_length
                    end_norm = read.reference_end / transcript_length if read.reference_end else start_norm + read.query_length / transcript_length
                    
                    # Skip invalid positions
                    if start_norm < 0 or end_norm > 1 or start_norm >= end_norm:
                        skipped_reads += 1
                        continue
                        
                    # Adjust for strand
                    if transcripts[transcript_id]['strand'] == '-':
                        start_norm, end_norm = 1 - end_norm, 1 - start_norm
                        
                    # Add to overall position distribution
                    positions.append((start_norm, end_norm, transcript_length))
                    
                    # Track by transcript
                    if transcript_id not in transcript_reads:
                        transcript_reads[transcript_id] = []
                    transcript_reads[transcript_id].append((start_norm, end_norm))
                    
                except Exception as e:
                    logger.warning(f"Error processing read: {str(e)}")
                    skipped_reads += 1
                    continue
    except Exception as e:
        logger.error(f"Error reading BAM file: {str(e)}")
        logger.warning("Using default coverage model instead.")
        if self.model_type == "10x_cdna":
            self._init_default_10x_cdna_model()
        else:
            self._init_default_direct_rna_model()
        return self
     
    logger.info(f"Processed {total_reads} total reads")
    logger.info(f"Skipped {skipped_reads} reads during processing")
    logger.info(f"Retained {len(positions)} valid read positions")
    
    if not positions:
        logger.error("No valid read positions found. Using default coverage model.")
        if self.model_type == "10x_cdna":
            self._init_default_10x_cdna_model()
        else:
            self._init_default_direct_rna_model()
        return self
        
    # 3. Filter transcripts with too few reads
    filtered_transcripts = {t: reads for t, reads in transcript_reads.items() 
                          if len(reads) >= min_reads}
    
    logger.info(f"Found {len(filtered_transcripts)} transcripts with >= {min_reads} reads")
    
    # 4. Create overall position distribution
    self.position_distribution = self._create_position_density(
        [p[0] for p in positions]  # Use start positions
    )
    
    # 5. Create length-dependent distributions
    # Group transcripts by length
    transcript_lengths = np.array([t['length'] for t in transcripts.values()])
    length_percentiles = np.percentile(transcript_lengths, 
                                     np.linspace(0, 100, length_bins+1))
    self.length_bins = length_percentiles
    
    # Create distribution for each length bin
    for i in range(length_bins):
        min_len = length_percentiles[i]
        max_len = length_percentiles[i+1]
        
        # Filter positions in this length bin
        bin_positions = [p[0] for p in positions 
                       if min_len <= p[2] < max_len]
        
        if len(bin_positions) > min_reads:
            self.length_dependent_distributions[i] = self._create_position_density(bin_positions)
            logger.info(f"Created distribution for length bin {i+1} ({min_len:.0f}-{max_len:.0f} bp) with {len(bin_positions)} positions")
        else:
            logger.warning(f"Insufficient positions ({len(bin_positions)}) for length bin {i+1} ({min_len:.0f}-{max_len:.0f} bp)")
    
    return self
    
def _prepare_bam_file(self, bam_file):
    """
    Prepare BAM file for coverage analysis by checking, sorting, and indexing.
    
    Args:
        bam_file: Path to input BAM file
        
    Returns:
        str: Path to prepared BAM file, or None if preparation failed
    """
    import tempfile
    import subprocess
    import os
    import shutil
    
    if not os.path.exists(bam_file):
        logger.error(f"BAM file does not exist: {bam_file}")
        return None
        
    # Create a temporary directory for processing
    temp_dir = tempfile.mkdtemp(prefix="coverage_model_")
    logger.info(f"Created temporary directory for BAM processing: {temp_dir}")
    
    try:
        # Check if BAM file is sorted
        is_sorted = False
        try:
            # Check BAM header for SO:coordinate
            header_cmd = f"samtools view -H {bam_file}"
            header_output = subprocess.check_output(header_cmd, shell=True, stderr=subprocess.STDOUT).decode('utf-8')
            is_sorted = "SO:coordinate" in header_output
        except subprocess.CalledProcessError:
            is_sorted = False
            
        # Sort BAM file if needed
        sorted_bam = os.path.join(temp_dir, "sorted.bam")
        if not is_sorted:
            logger.info(f"Sorting BAM file: {bam_file}")
            sort_cmd = f"samtools sort -o {sorted_bam} {bam_file}"
            try:
                subprocess.check_call(sort_cmd, shell=True, stderr=subprocess.STDOUT)
                logger.info(f"Successfully sorted BAM file to: {sorted_bam}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to sort BAM file: {str(e)}")
                shutil.rmtree(temp_dir)
                return None
        else:
            # If already sorted, just copy the file
            shutil.copy(bam_file, sorted_bam)
            logger.info(f"BAM file already sorted: {bam_file}")
            
        # Index the sorted BAM file
        logger.info(f"Indexing BAM file: {sorted_bam}")
        index_cmd = f"samtools index {sorted_bam}"
        try:
            subprocess.check_call(index_cmd, shell=True, stderr=subprocess.STDOUT)
            logger.info(f"Successfully indexed BAM file: {sorted_bam}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to index BAM file: {str(e)}")
            logger.warning("Proceeding without index, but performance may be affected")
            
        return sorted_bam
        
    except Exception as e:
        logger.error(f"Error preparing BAM file: {str(e)}")
        shutil.rmtree(temp_dir)
        return None 

def _create_position_density(self, positions):
    """
    Create a position density distribution from a list of positions.
    
    Args:
        positions: List of normalized positions (0-1)
        
    Returns:
        tuple: (x_values, y_values) for the density distribution
    """
    if not positions:
        logger.warning("No positions provided for density estimation. Using uniform distribution.")
        x = np.linspace(0, 1, 1000)
        y = np.ones_like(x) / len(x)
        return (x, y)
        
    positions = np.array(positions)
    
    # Remove any NaN or infinite values
    positions = positions[~np.isnan(positions) & ~np.isinf(positions)]
    
    if len(positions) < 2:
        logger.warning("Insufficient positions for density estimation. Using uniform distribution.")
        x = np.linspace(0, 1, 1000)
        y = np.ones_like(x) / len(x)
        return (x, y)
        
    try:
        # Create kernel density estimate
        kde = stats.gaussian_kde(positions, bw_method=self.smoothing_factor)
        
        # Evaluate on a grid
        x = np.linspace(0, 1, 1000)
        y = kde(x)
        
        # Normalize
        y = y / np.sum(y)
        
        return (x, y)
        
    except Exception as e:
        logger.warning(f"Error creating density estimate: {str(e)}. Using uniform distribution.")
        x = np.linspace(0, 1, 1000)
        y = np.ones_like(x) / len(x)
        return (x, y)

def _sample_from_distribution(self, distribution):
    """
    Sample a value from a probability distribution.
    
    Args:
        distribution: Tuple of (x_values, probabilities)
        
    Returns:
        float: Sampled value from the distribution
    """
    x, y = distribution
    return np.random.choice(x, p=y/np.sum(y))

def _parse_annotation(self, annotation_file):
    """
    Parse annotation file to get transcript information.
    
    Args:
        annotation_file: Path to GTF/GFF annotation file
        
    Returns:
        dict: Dictionary of transcript information
    """
    transcripts = {}
    
    try:
        with open(annotation_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                    
                if fields[2].lower() in ('transcript', 'mrna'):
                    # Try to extract transcript ID from attributes field
                    attributes = fields[8]
                    transcript_id = None
                    
                    # Try different formats
                    # Format 1: transcript_id "ENST00000456328.2";
                    if 'transcript_id' in attributes:
                        try:
                            transcript_id = attributes.split('transcript_id')[1].split(';')[0].strip().strip('"\'')
                        except:
                            pass
                    
                    # Format 2: ID=transcript:ENST00000456328.2;
                    if transcript_id is None and 'ID=' in attributes:
                        try:
                            id_part = [p for p in attributes.split(';') if p.strip().startswith('ID=')]
                            if id_part:
                                transcript_id = id_part[0].split('=')[1].strip().strip('"\'')
                                if ':' in transcript_id:
                                    transcript_id = transcript_id.split(':')[1]
                        except:
                            pass
                    
                    # Format 3: Parent=gene:ENSG00000223972.5;ID=ENST00000456328.2;
                    if transcript_id is None and 'ID=' in attributes:
                        try:
                            id_part = [p for p in attributes.split(';') if p.strip().startswith('ID=')]
                            if id_part:
                                transcript_id = id_part[0].split('=')[1].strip().strip('"\'')
                        except:
                            pass
                            
                    # Simple format: ID=ENST00000456328.2
                    if transcript_id is None and attributes.startswith('ID='):
                        try:
                            transcript_id = attributes.split('=')[1].split(';')[0].strip().strip('"\'')
                        except:
                            pass
                    
                    # If we found a transcript ID and it's valid
                    if transcript_id:
                        # Get strand
                        strand = fields[6]
                        
                        # Get length (end - start + 1)
                        try:
                            start = int(fields[3])
                            end = int(fields[4])
                            length = end - start + 1
                        except:
                            # If we can't calculate length, skip this transcript
                            continue
                        
                        transcripts[transcript_id] = {'strand': strand, 'length': length}
    except Exception as e:
        logger.warning(f"Error parsing annotation file: {str(e)}")
        
    if not transcripts:
        logger.warning("No transcripts found in annotation file. Using fallback parsing method.")
        
        try:
            # Fallback to simple parsing
            with open(annotation_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    fields = line.strip().split('\t')
                    if len(fields) < 8:
                        continue
                        
                    # Just use the first column as ID and assume it's a transcript
                    transcript_id = fields[0]
                    strand = fields[6] if len(fields) > 6 else '+'
                    try:
                        start = int(fields[3]) if len(fields) > 3 else 1
                        end = int(fields[4]) if len(fields) > 4 else 1000
                        length = end - start + 1
                    except:
                        length = 1000  # Default length
                    
                    transcripts[transcript_id] = {'strand': strand, 'length': length}
        except Exception as e:
            logger.error(f"Fallback parsing also failed: {str(e)}")
    
    logger.info(f"Parsed {len(transcripts)} transcripts from annotation")
    return transcripts 