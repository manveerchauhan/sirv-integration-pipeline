"""
Unit tests for the mapping module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import pandas as pd

# Add parent directory to path for importing the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sirv_pipeline.mapping import (
    map_sirv_reads,
    get_transcript_statistics,
    _parse_transcripts_from_gtf,
    _assign_transcripts_to_reads
)


class TestMapping(unittest.TestCase):
    """Test case for mapping module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock GTF file
        self.mock_gtf = os.path.join(self.temp_dir.name, "mock.gtf")
        with open(self.mock_gtf, 'w') as f:
            f.write('# Comment line\n')
            f.write('SIRV1\tENSEMBL\ttranscript\t1\t1000\t.\t+\t.\ttranscript_id "SIRV101"; gene_id "SIRV1";\n')
            f.write('SIRV1\tENSEMBL\texon\t1\t100\t.\t+\t.\ttranscript_id "SIRV101"; gene_id "SIRV1";\n')
            f.write('SIRV2\tENSEMBL\ttranscript\t1\t2000\t.\t+\t.\ttranscript_id "SIRV201"; gene_id "SIRV2";\n')
            f.write('SIRV2\tENSEMBL\texon\t1\t200\t.\t+\t.\ttranscript_id "SIRV201"; gene_id "SIRV2";\n')
        
        # Create mock mapping CSV
        self.mock_csv = os.path.join(self.temp_dir.name, "mock_map.csv")
        mock_data = {
            'read_id': ['read1', 'read2', 'read3', 'read4'],
            'sirv_transcript': ['SIRV101', 'SIRV101', 'SIRV201', 'SIRV201'],
            'overlap_fraction': [0.8, 0.9, 0.7, 0.6],
            'read_length': [500, 600, 700, 800],
            'alignment_length': [400, 550, 650, 750]
        }
        pd.DataFrame(mock_data).to_csv(self.mock_csv, index=False)
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.temp_dir.cleanup()
    
    def test_parse_transcripts_from_gtf(self):
        """Test parsing transcripts from GTF file"""
        transcripts = _parse_transcripts_from_gtf(self.mock_gtf)
        
        # Check results
        self.assertEqual(len(transcripts), 2)
        self.assertIn('SIRV101', transcripts)
        self.assertIn('SIRV201', transcripts)
        
        # Check transcript properties
        self.assertEqual(transcripts['SIRV101']['chrom'], 'SIRV1')
        self.assertEqual(transcripts['SIRV101']['start'], 1)
        self.assertEqual(transcripts['SIRV101']['end'], 1000)
        self.assertEqual(transcripts['SIRV101']['strand'], '+')
        
        self.assertEqual(transcripts['SIRV201']['chrom'], 'SIRV2')
        self.assertEqual(transcripts['SIRV201']['start'], 1)
        self.assertEqual(transcripts['SIRV201']['end'], 2000)
        self.assertEqual(transcripts['SIRV201']['strand'], '+')
    
    @patch('sirv_pipeline.mapping.pysam.AlignmentFile')
    def test_assign_transcripts_to_reads(self, mock_pysam):
        """Test assigning transcripts to reads"""
        # Mock aligned reads
        mock_read1 = MagicMock()
        mock_read1.is_unmapped = False
        mock_read1.query_name = 'read1'
        mock_read1.reference_name = 'SIRV1'
        mock_read1.reference_start = 50
        mock_read1.reference_end = 950
        mock_read1.query_length = 900
        
        mock_read2 = MagicMock()
        mock_read2.is_unmapped = False
        mock_read2.query_name = 'read2'
        mock_read2.reference_name = 'SIRV2'
        mock_read2.reference_start = 100
        mock_read2.reference_end = 800
        mock_read2.query_length = 700
        
        mock_bam = MagicMock()
        mock_bam.__enter__.return_value = mock_bam
        mock_bam.__exit__.return_value = None
        mock_bam.fetch.return_value = [mock_read1, mock_read2]
        
        mock_pysam.return_value = mock_bam
        
        # Create transcript dictionary
        transcripts = {
            'SIRV101': {'chrom': 'SIRV1', 'start': 1, 'end': 1000, 'strand': '+'},
            'SIRV201': {'chrom': 'SIRV2', 'start': 1, 'end': 2000, 'strand': '+'}
        }
        
        # Test function
        mappings = _assign_transcripts_to_reads('mock.bam', transcripts, min_overlap=0.5)
        
        # Check results
        self.assertEqual(len(mappings), 2)
        
        # Check first mapping
        self.assertEqual(mappings[0]['read_id'], 'read1')
        self.assertEqual(mappings[0]['sirv_transcript'], 'SIRV101')
        self.assertGreaterEqual(mappings[0]['overlap_fraction'], 0.5)
        
        # Check second mapping
        self.assertEqual(mappings[1]['read_id'], 'read2')
        self.assertEqual(mappings[1]['sirv_transcript'], 'SIRV201')
        self.assertGreaterEqual(mappings[1]['overlap_fraction'], 0.5)
    
    def test_get_transcript_statistics(self):
        """Test getting transcript statistics"""
        stats = get_transcript_statistics(self.mock_csv)
        
        # Check results
        self.assertEqual(stats['total_reads'], 4)
        self.assertEqual(stats['unique_transcripts'], 2)
        self.assertIn('SIRV101', stats['reads_per_transcript'])
        self.assertIn('SIRV201', stats['reads_per_transcript'])
        self.assertEqual(stats['reads_per_transcript']['SIRV101'], 2)
        self.assertEqual(stats['reads_per_transcript']['SIRV201'], 2)
    
    @patch('sirv_pipeline.mapping.subprocess.run')
    @patch('sirv_pipeline.mapping._parse_transcripts_from_gtf')
    @patch('sirv_pipeline.mapping._assign_transcripts_to_reads')
    def test_map_sirv_reads(self, mock_assign, mock_parse, mock_subprocess):
        """Test mapping SIRV reads to reference"""
        # Mock subprocess.run
        mock_subprocess.return_value = MagicMock()
        
        # Mock _parse_transcripts_from_gtf
        mock_parse.return_value = {
            'SIRV101': {'chrom': 'SIRV1', 'start': 1, 'end': 1000, 'strand': '+'},
            'SIRV201': {'chrom': 'SIRV2', 'start': 1, 'end': 2000, 'strand': '+'}
        }
        
        # Mock _assign_transcripts_to_reads
        mock_assign.return_value = [
            {'read_id': 'read1', 'sirv_transcript': 'SIRV101'},
            {'read_id': 'read2', 'sirv_transcript': 'SIRV101'},
            {'read_id': 'read3', 'sirv_transcript': 'SIRV201'},
            {'read_id': 'read4', 'sirv_transcript': 'SIRV201'}
        ]
        
        # Test function
        sirv_fastq = os.path.join(self.temp_dir.name, "sirv.fastq")
        sirv_reference = os.path.join(self.temp_dir.name, "sirv.fa")
        output_csv = os.path.join(self.temp_dir.name, "output.csv")
        
        # Create mock input files
        with open(sirv_fastq, 'w') as f:
            f.write("dummy content")
        
        with open(sirv_reference, 'w') as f:
            f.write("dummy content")
        
        # Run function
        result = map_sirv_reads(
            sirv_fastq=sirv_fastq,
            sirv_reference=sirv_reference,
            sirv_gtf=self.mock_gtf,
            output_csv=output_csv,
            threads=4,
            min_overlap=0.5
        )
        
        # Check results
        self.assertEqual(result, output_csv)
        self.assertTrue(os.path.exists(output_csv))
        
        # Check subprocess calls
        # Align with minimap2
        self.assertEqual(mock_subprocess.call_count, 3)
        
        # Check if _parse_transcripts_from_gtf was called
        mock_parse.assert_called_once_with(self.mock_gtf)
        
        # Check if _assign_transcripts_to_reads was called
        self.assertEqual(mock_assign.call_count, 1)
        
        # Verify output CSV content
        df = pd.read_csv(output_csv)
        self.assertEqual(len(df), 4)
        self.assertEqual(list(df.columns), ['read_id', 'sirv_transcript'])
        self.assertEqual(list(df['read_id']), ['read1', 'read2', 'read3', 'read4'])
        self.assertEqual(list(df['sirv_transcript']), ['SIRV101', 'SIRV101', 'SIRV201', 'SIRV201'])


if __name__ == '__main__':
    unittest.main()