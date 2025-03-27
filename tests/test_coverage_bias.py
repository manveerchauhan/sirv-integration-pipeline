"""
Unit tests for the CoverageBiasModel class.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import numpy as np
import pandas as pd

# Add parent directory to path for importing the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sirv_pipeline.coverage_bias import CoverageBiasModel


class TestCoverageBiasModel(unittest.TestCase):
    """Test case for CoverageBiasModel class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create coverage bias model with fixed seed
        self.model = CoverageBiasModel(bins=10, seed=42)
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test initializing the CoverageBiasModel"""
        # Default initialization
        model = CoverageBiasModel()
        self.assertEqual(model.bins, 100)
        self.assertEqual(model.smoothing, 0.1)
        self.assertEqual(model.min_coverage, 5)
        self.assertFalse(model.has_model)
        self.assertIsNone(model.start_dist)
        self.assertIsNone(model.end_dist)
        
        # Custom initialization
        model = CoverageBiasModel(bins=50, smoothing=0.2, min_coverage=10, seed=123)
        self.assertEqual(model.bins, 50)
        self.assertEqual(model.smoothing, 0.2)
        self.assertEqual(model.min_coverage, 10)
    
    @patch('sirv_pipeline.coverage_bias.pysam.AlignmentFile')
    @patch('sirv_pipeline.coverage_bias.SeqIO')
    def test_learn_from_bam(self, mock_seqio, mock_pysam):
        """Test learning coverage bias from BAM file"""
        # Mock SeqIO for reference sequence parsing
        mock_record1 = MagicMock()
        mock_record1.id = "transcript1"
        mock_record1.seq = "A" * 1000
        
        mock_record2 = MagicMock()
        mock_record2.id = "transcript2"
        mock_record2.seq = "G" * 2000
        
        mock_seqio.parse.return_value = [mock_record1, mock_record2]
        
        # Mock aligned reads
        mock_read1 = MagicMock()
        mock_read1.is_unmapped = False
        mock_read1.is_secondary = False
        mock_read1.is_supplementary = False
        mock_read1.reference_name = "transcript1"
        mock_read1.reference_start = 0  # 5' end
        mock_read1.reference_end = 800
        
        mock_read2 = MagicMock()
        mock_read2.is_unmapped = False
        mock_read2.is_secondary = False
        mock_read2.is_supplementary = False
        mock_read2.reference_name = "transcript1"
        mock_read2.reference_start = 100  # Middle
        mock_read2.reference_end = 900
        
        mock_read3 = MagicMock()
        mock_read3.is_unmapped = False
        mock_read3.is_secondary = False
        mock_read3.is_supplementary = False
        mock_read3.reference_name = "transcript2"
        mock_read3.reference_start = 1500  # 3' end
        mock_read3.reference_end = 2000
        
        # Configure mock AlignmentFile
        mock_bam = MagicMock()
        mock_bam.__enter__.return_value = mock_bam
        mock_bam.__exit__.return_value = None
        mock_bam.fetch.return_value = [mock_read1, mock_read2, mock_read3]
        
        mock_pysam.return_value = mock_bam
        
        # Test learn_from_bam
        bam_file = os.path.join(self.temp_dir.name, "test.bam")
        ref_file = os.path.join(self.temp_dir.name, "reference.fa")
        
        # Create mock files
        with open(bam_file, 'w') as f:
            f.write("dummy")
        
        with open(ref_file, 'w') as f:
            f.write("dummy")
        
        # Run the function
        result = self.model.learn_from_bam(bam_file, ref_file)
        
        # Check results
        self.assertTrue(result)
        self.assertTrue(self.model.has_model)
        self.assertIsNotNone(self.model.start_dist)
        self.assertIsNotNone(self.model.end_dist)
        self.assertEqual(len(self.model.start_dist), 10)  # bins=10
        self.assertEqual(len(self.model.end_dist), 10)
        
        # Check that distributions sum to 1
        self.assertAlmostEqual(np.sum(self.model.start_dist), 1.0, places=5)
        self.assertAlmostEqual(np.sum(self.model.end_dist), 1.0, places=5)
    
    def test_learn_from_fastq(self):
        """Test learning from FASTQ (simplified method)"""
        # Run the function with minimal inputs
        result = self.model.learn_from_fastq("dummy.fastq")
        
        # Even without transcript info, it should create a synthetic model
        self.assertTrue(result)
        self.assertTrue(self.model.has_model)
        self.assertIsNotNone(self.model.start_dist)
        self.assertIsNotNone(self.model.end_dist)
        
        # Distributions should sum to 1
        self.assertAlmostEqual(np.sum(self.model.start_dist), 1.0, places=5)
        self.assertAlmostEqual(np.sum(self.model.end_dist), 1.0, places=5)
        
        # 5' bias for starts (higher at beginning)
        self.assertGreater(self.model.start_dist[0], self.model.start_dist[-1])
        
        # 3' bias for ends (higher at end)
        self.assertGreater(self.model.end_dist[-1], self.model.end_dist[0])
    
    @patch('sirv_pipeline.coverage_bias.plt')
    def test_plot_distributions(self, mock_plt):
        """Test plotting distributions"""
        # We need a model first
        self.model.learn_from_fastq("dummy.fastq")
        
        # Test plotting
        plot_file = os.path.join(self.temp_dir.name, "coverage_bias.png")
        result = self.model.plot_distributions(plot_file)
        
        # Check results
        self.assertTrue(result)
        
        # Check matplotlib calls
        mock_plt.figure.assert_called_once()
        mock_plt.savefig.assert_called_once_with(plot_file, dpi=300)
        mock_plt.close.assert_called_once()
        
        # Test without a model
        model = CoverageBiasModel()
        result = model.plot_distributions(plot_file)
        self.assertFalse(result)
    
    def test_sample_read_position(self):
        """Test sampling read positions"""
        # We need a model first
        self.model.learn_from_fastq("dummy.fastq")
        
        # Test sampling multiple positions
        transcript_length = 1000
        positions = []
        
        for _ in range(100):
            start, end = self.model.sample_read_position(transcript_length)
            
            # Check constraints
            self.assertGreaterEqual(start, 0)
            self.assertLess(start, transcript_length)
            self.assertGreater(end, start)
            self.assertLessEqual(end, transcript_length)
            
            positions.append((start, end))
        
        # Check distribution statistics
        starts = [p[0] for p in positions]
        ends = [p[1] for p in positions]
        
        # With 5' bias, more reads should start near the beginning
        start_first_quarter = sum(1 for s in starts if s < transcript_length/4)
        start_last_quarter = sum(1 for s in starts if s > 3*transcript_length/4)
        self.assertGreater(start_first_quarter, start_last_quarter)
        
        # With 3' bias, more reads should end near the end
        end_first_quarter = sum(1 for e in ends if e < transcript_length/4)
        end_last_quarter = sum(1 for e in ends if e > 3*transcript_length/4)
        self.assertGreater(end_last_quarter, end_first_quarter)
        
        # Test without a model (should use default behavior)
        model = CoverageBiasModel()
        start, end = model.sample_read_position(transcript_length)
        self.assertGreaterEqual(start, 0)
        self.assertLessEqual(end, transcript_length)
    
    def test_apply_to_sequence(self):
        """Test applying coverage bias to a sequence"""
        # We need a model first
        self.model.learn_from_fastq("dummy.fastq")
        
        # Test sequence
        sequence = "A" * 1000
        quality = "I" * 1000
        
        # Apply without target length
        result_seq, result_qual = self.model.apply_to_sequence(sequence, quality)
        
        # Check results
        self.assertLessEqual(len(result_seq), len(sequence))
        self.assertEqual(len(result_seq), len(result_qual))
        self.assertTrue(all(base == 'A' for base in result_seq))
        self.assertTrue(all(qual == 'I' for qual in result_qual))
        
        # Apply with target length
        target_length = 500
        result_seq, result_qual = self.model.apply_to_sequence(sequence, quality, target_length)
        
        # Check results with target length
        self.assertLessEqual(len(result_seq), target_length)
        self.assertEqual(len(result_seq), len(result_qual))
        
        # Test with short sequence (should return unchanged)
        short_seq = "ACGT"
        short_qual = "IIII"
        result_seq, result_qual = self.model.apply_to_sequence(short_seq, short_qual)
        self.assertEqual(result_seq, short_seq)
        self.assertEqual(result_qual, short_qual)
    
    def test_save_and_load(self):
        """Test saving and loading the model"""
        # Create and train a model
        self.model.learn_from_fastq("dummy.fastq")
        
        # Save the model
        model_file = os.path.join(self.temp_dir.name, "coverage_model.pkl")
        result = self.model.save(model_file)
        
        # Check result
        self.assertTrue(result)
        self.assertTrue(os.path.exists(model_file))
        
        # Load the model
        loaded_model = CoverageBiasModel.load(model_file)
        
        # Check loaded model
        self.assertTrue(loaded_model.has_model)
        self.assertEqual(loaded_model.bins, self.model.bins)
        self.assertEqual(loaded_model.smoothing, self.model.smoothing)
        self.assertEqual(loaded_model.min_coverage, self.model.min_coverage)
        
        np.testing.assert_array_almost_equal(loaded_model.start_dist, self.model.start_dist)
        np.testing.assert_array_almost_equal(loaded_model.end_dist, self.model.end_dist)
        
        # Test saving without a model
        model = CoverageBiasModel()
        result = model.save(model_file)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()