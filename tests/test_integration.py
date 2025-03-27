"""
Unit tests for the integration module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import numpy as np
import pandas as pd
from io import StringIO

# Add parent directory to path for importing the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sirv_pipeline.integration import (
    CellBarcode,
    ReadLengthSampler,
    extract_cell_info,
    add_sirv_to_dataset,
    _load_sirv_reads,
    _copy_fastq_contents
)


class TestCellBarcode(unittest.TestCase):
    """Test case for CellBarcode class"""
    
    def test_generate_barcodes(self):
        """Test generating cell barcodes"""
        # Test with fixed seed for reproducibility
        barcode_gen = CellBarcode(seed=42)
        barcodes = barcode_gen.generate_barcodes(5)
        
        # Check results
        self.assertEqual(len(barcodes), 5)
        for barcode in barcodes:
            self.assertEqual(len(barcode), 16)
            for base in barcode:
                self.assertIn(base, ['A', 'C', 'G', 'T'])
    
    def test_generate_umi(self):
        """Test generating UMIs"""
        barcode_gen = CellBarcode(seed=42)
        
        # Test default length
        umi = barcode_gen.generate_umi()
        self.assertEqual(len(umi), 12)
        for base in umi:
            self.assertIn(base, ['A', 'C', 'G', 'T'])
        
        # Test custom length
        umi = barcode_gen.generate_umi(length=8)
        self.assertEqual(len(umi), 8)
        for base in umi:
            self.assertIn(base, ['A', 'C', 'G', 'T'])


class TestReadLengthSampler(unittest.TestCase):
    """Test case for ReadLengthSampler class"""
    
    def test_initialization(self):
        """Test initializing ReadLengthSampler"""
        # Default initialization
        sampler = ReadLengthSampler()
        self.assertIsNone(sampler.length_distribution)
        self.assertEqual(sampler.default_mean, 1000)
        self.assertEqual(sampler.default_std, 300)
        self.assertEqual(sampler.min_length, 100)
        
        # Custom initialization
        sampler = ReadLengthSampler(
            default_mean=500,
            default_std=100,
            min_length=50
        )
        self.assertEqual(sampler.default_mean, 500)
        self.assertEqual(sampler.default_std, 100)
        self.assertEqual(sampler.min_length, 50)
    
    def test_generate_default_distribution(self):
        """Test generating default distribution"""
        sampler = ReadLengthSampler(
            default_mean=500,
            default_std=100,
            min_length=50
        )
        
        # Generate distribution
        lengths = sampler._generate_default_distribution(size=1000)
        
        # Check results
        self.assertEqual(len(lengths), 1000)
        self.assertTrue(all(length >= 50 for length in lengths))
        self.assertTrue(400 < np.mean(lengths) < 600)  # Approximate mean
    
    @patch('sirv_pipeline.integration.SeqIO')
    def test_sample_from_fastq(self, mock_seqio):
        """Test sampling read lengths from FASTQ"""
        # Mock sequences
        mock_record1 = MagicMock()
        mock_record1.seq = 'A' * 1000
        
        mock_record2 = MagicMock()
        mock_record2.seq = 'A' * 1500
        
        mock_record3 = MagicMock()
        mock_record3.seq = 'A' * 2000
        
        # Configure mock to return our mock records
        mock_seqio.parse.return_value = [mock_record1, mock_record2, mock_record3]
        
        # Test function
        sampler = ReadLengthSampler()
        lengths = sampler.sample_from_fastq('mock.fastq', sample_size=3)
        
        # Check results
        self.assertEqual(len(lengths), 3)
        self.assertEqual(lengths[0], 1000)
        self.assertEqual(lengths[1], 1500)
        self.assertEqual(lengths[2], 2000)
        self.assertEqual(sampler.length_distribution.tolist(), [1000, 1500, 2000])
    
    def test_sample(self):
        """Test sampling read lengths"""
        # Test with predefined distribution
        sampler = ReadLengthSampler(length_distribution=np.array([1000, 2000, 3000]))
        length = sampler.sample()
        self.assertIn(length, [1000, 2000, 3000])
        
        # Test with default distribution
        sampler = ReadLengthSampler()
        length = sampler.sample()
        self.assertIsInstance(length, int)
        self.assertGreaterEqual(length, 100)


class TestIntegrationFunctions(unittest.TestCase):
    """Test case for integration module functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock mapping CSV
        self.mock_csv = os.path.join(self.temp_dir.name, "mock_map.csv")
        mock_data = {
            'read_id': ['read1', 'read2', 'read3', 'read4'],
            'sirv_transcript': ['SIRV101', 'SIRV101', 'SIRV201', 'SIRV201']
        }
        pd.DataFrame(mock_data).to_csv(self.mock_csv, index=False)
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.temp_dir.cleanup()
    
    @patch('sirv_pipeline.integration.CellBarcode')
    def test_extract_cell_info(self, mock_cell_barcode):
        """Test extracting cell info from scRNA-seq data"""
        # Mock CellBarcode.generate_barcodes
        mock_cell_barcode.return_value.generate_barcodes.return_value = [
            'CELL1', 'CELL2', 'CELL3', 'CELL4', 'CELL5',
            'CELL6', 'CELL7', 'CELL8', 'CELL9', 'CELL10'
        ]
        
        # Test with fixed seed for reproducibility
        with patch('numpy.random.randint', return_value=2000):
            cell_info = extract_cell_info('mock.fastq')
            
            # Check results
            self.assertEqual(len(cell_info), 10)
            for cell_id, count in cell_info.items():
                self.assertEqual(count, 2000)
    
    @patch('sirv_pipeline.integration.SeqIO')
    def test_load_sirv_reads(self, mock_seqio):
        """Test loading SIRV reads"""
        # Mock sequences
        mock_record1 = MagicMock()
        mock_record1.id = 'read1'
        mock_record1.seq = 'ACGT' * 250  # 1000 bp
        mock_record1.letter_annotations = {'phred_quality': [30] * 1000}
        
        mock_record2 = MagicMock()
        mock_record2.id = 'read2'
        mock_record2.seq = 'ACGT' * 375  # 1500 bp
        mock_record2.letter_annotations = {'phred_quality': [30] * 1500}
        
        # Unknown read (not in mapping)
        mock_record3 = MagicMock()
        mock_record3.id = 'unknown'
        mock_record3.seq = 'ACGT' * 500  # 2000 bp
        mock_record3.letter_annotations = {'phred_quality': [30] * 2000}
        
        # Configure mock to return our mock records
        mock_seqio.parse.return_value = [mock_record1, mock_record2, mock_record3]
        
        # Create read to transcript mapping
        read_to_transcript = {
            'read1': 'SIRV101',
            'read2': 'SIRV101'
        }
        
        # Test function
        sirv_reads = _load_sirv_reads('mock.fastq', read_to_transcript)
        
        # Check results
        self.assertEqual(len(sirv_reads), 2)
        self.assertIn('read1', sirv_reads)
        self.assertIn('read2', sirv_reads)
        self.assertNotIn('unknown', sirv_reads)
        
        # Check read properties
        self.assertEqual(sirv_reads['read1']['transcript'], 'SIRV101')
        self.assertEqual(len(sirv_reads['read1']['seq']), 1000)
        self.assertEqual(len(sirv_reads['read1']['qual']), 1000)
        
        self.assertEqual(sirv_reads['read2']['transcript'], 'SIRV101')
        self.assertEqual(len(sirv_reads['read2']['seq']), 1500)
        self.assertEqual(len(sirv_reads['read2']['qual']), 1500)
    
    def test_copy_fastq_contents(self):
        """Test copying FASTQ contents"""
        # Mock FASTQ file content
        mock_fastq_content = (
            "@read1\n"
            "ACGT\n"
            "+\n"
            "IIII\n"
            "@read2\n"
            "ACGT\n"
            "+\n"
            "IIII\n"
        )
        
        # Mock output file handle
        mock_out = StringIO()
        
        # Test function with mock open
        with patch('builtins.open', mock_open(read_data=mock_fastq_content)):
            read_count = _copy_fastq_contents('mock.fastq', mock_out)
        
        # Check results
        self.assertEqual(read_count, 2)
        self.assertEqual(mock_out.getvalue(), mock_fastq_content)
    
    @patch('sirv_pipeline.integration.extract_cell_info')
    @patch('sirv_pipeline.integration._load_sirv_reads')
    @patch('sirv_pipeline.integration._copy_fastq_contents')
    @patch('sirv_pipeline.integration.open')
    @patch('sirv_pipeline.integration.ReadLengthSampler')
    @patch('sirv_pipeline.integration.CellBarcode')
    def test_add_sirv_to_dataset(self, mock_barcode, mock_sampler, mock_open, 
                               mock_copy, mock_load, mock_extract):
        """Test adding SIRV reads to dataset"""
        # Mock extract_cell_info
        mock_extract.return_value = {
            'CELL1': 1000,
            'CELL2': 2000
        }
        
        # Mock _load_sirv_reads
        mock_sirv_reads = {
            'read1': {
                'seq': 'A' * 1000,
                'qual': 'I' * 1000,
                'transcript': 'SIRV101'
            },
            'read2': {
                'seq': 'A' * 1500,
                'qual': 'I' * 1500,
                'transcript': 'SIRV101'
            },
            'read3': {
                'seq': 'A' * 2000,
                'qual': 'I' * 2000,
                'transcript': 'SIRV201'
            }
        }
        mock_load.return_value = mock_sirv_reads
        
        # Mock _copy_fastq_contents
        mock_copy.return_value = 3000  # 3000 reads copied
        
        # Mock open
        mock_outfile = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_outfile
        
        # Mock ReadLengthSampler
        mock_sampler.return_value.sample_from_fastq.return_value = np.array([1000, 1500, 2000])
        mock_sampler.return_value.sample.return_value = 1000
        
        # Mock CellBarcode
        mock_barcode.return_value.generate_umi.return_value = 'ACGTACGTAC'
        
        # Test function
        sirv_fastq = os.path.join(self.temp_dir.name, "sirv.fastq")
        sc_fastq = os.path.join(self.temp_dir.name, "sc.fastq")
        output_fastq = os.path.join(self.temp_dir.name, "output.fastq")
        
        # Create mock input files
        with open(sirv_fastq, 'w') as f:
            f.write("dummy content")
        
        with open(sc_fastq, 'w') as f:
            f.write("dummy content")
        
        # Run function with fixed seed
        with patch('random.sample', return_value=['read1', 'read2']):
            with patch('random.choice', return_value='read1'):
                output_fastq, tracking_file, expected_file = add_sirv_to_dataset(
                    sirv_fastq=sirv_fastq,
                    sc_fastq=sc_fastq,
                    sirv_map_csv=self.mock_csv,
                    output_fastq=output_fastq,
                    insertion_rate=0.01,
                    seed=42
                )
        
        # Check results
        self.assertEqual(output_fastq, os.path.join(self.temp_dir.name, "output.fastq"))
        self.assertEqual(tracking_file, os.path.join(self.temp_dir.name, "output_tracking.csv"))
        self.assertEqual(expected_file, os.path.join(self.temp_dir.name, "output_expected_counts.csv"))
        
        # Check function calls
        mock_extract.assert_called_once_with(sc_fastq)
        mock_load.assert_called_once_with(sirv_fastq, {'read1': 'SIRV101', 'read2': 'SIRV101', 'read3': 'SIRV201', 'read4': 'SIRV201'})
        mock_sampler.return_value.sample_from_fastq.assert_called_once_with(sc_fastq, sample_size=1000)
        mock_copy.assert_called_once()
        
        # At least 30 writes to the output file (FASTQ records)
        self.assertGreaterEqual(mock_outfile.write.call_count, 30)


if __name__ == '__main__':
    unittest.main()