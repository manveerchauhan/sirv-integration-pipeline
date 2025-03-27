"""
Unit tests for the evaluation module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for importing the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sirv_pipeline.evaluation import (
    compare_with_flames,
    generate_report,
    _extract_sirv_counts,
    _generate_summary,
    _generate_plots
)


class TestEvaluation(unittest.TestCase):
    """Test case for evaluation module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock expected counts CSV
        self.expected_file = os.path.join(self.temp_dir.name, "expected.csv")
        expected_data = {
            'barcode': ['CELL1', 'CELL1', 'CELL2', 'CELL2'],
            'sirv_transcript': ['SIRV101', 'SIRV201', 'SIRV101', 'SIRV201'],
            'expected_count': [10, 5, 8, 4]
        }
        pd.DataFrame(expected_data).to_csv(self.expected_file, index=False)
        
        # Create mock FLAMES output CSV
        self.flames_file = os.path.join(self.temp_dir.name, "flames.csv")
        flames_data = {
            'cell_barcode': ['CELL1', 'CELL1', 'CELL2', 'CELL2', 'CELL1'],
            'transcript_id': ['SIRV101', 'SIRV201', 'SIRV101', 'ENST01', 'ENST02'],
            'count': [8, 4, 6, 10, 5]
        }
        pd.DataFrame(flames_data).to_csv(self.flames_file, index=False)
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.temp_dir.cleanup()
    
    def test_extract_sirv_counts(self):
        """Test extracting SIRV counts from FLAMES output"""
        # Load FLAMES data
        flames_df = pd.read_csv(self.flames_file)
        
        # Test function
        sirv_counts = _extract_sirv_counts(flames_df)
        
        # Check results
        self.assertEqual(len(sirv_counts), 3)
        self.assertEqual(list(sirv_counts.columns), ['barcode', 'sirv_transcript', 'observed_count'])
        
        # Check SIRV counts
        self.assertEqual(sirv_counts.loc[0, 'barcode'], 'CELL1')
        self.assertEqual(sirv_counts.loc[0, 'sirv_transcript'], 'SIRV101')
        self.assertEqual(sirv_counts.loc[0, 'observed_count'], 8)
        
        self.assertEqual(sirv_counts.loc[1, 'barcode'], 'CELL1')
        self.assertEqual(sirv_counts.loc[1, 'sirv_transcript'], 'SIRV201')
        self.assertEqual(sirv_counts.loc[1, 'observed_count'], 4)
        
        self.assertEqual(sirv_counts.loc[2, 'barcode'], 'CELL2')
        self.assertEqual(sirv_counts.loc[2, 'sirv_transcript'], 'SIRV101')
        self.assertEqual(sirv_counts.loc[2, 'observed_count'], 6)
    
    def test_generate_summary(self):
        """Test generating summary statistics"""
        # Create comparison DataFrame
        comparison_data = {
            'barcode': ['CELL1', 'CELL1', 'CELL2', 'CELL2'],
            'sirv_transcript': ['SIRV101', 'SIRV201', 'SIRV101', 'SIRV201'],
            'expected_count': [10, 5, 8, 4],
            'observed_count': [8, 4, 6, 0],
            'detected': [True, True, True, False],
            'detection_rate': [0.8, 0.8, 0.75, 0.0]
        }
        comparison = pd.DataFrame(comparison_data)
        
        # Test function
        summary = _generate_summary(comparison)
        
        # Check results
        self.assertEqual(summary['total_expected'], 4)
        self.assertEqual(summary['total_detected'], 3)
        self.assertAlmostEqual(summary['detection_rate'], 0.75)
        
        # Check correlation
        expected_counts = comparison['expected_count'].values
        observed_counts = comparison['observed_count'].values
        expected_correlation = np.corrcoef(expected_counts, observed_counts)[0, 1]
        self.assertAlmostEqual(summary['correlation'], expected_correlation)
        
        # Check transcript metrics
        self.assertIn('transcript_metrics', summary)
        transcript_metrics = summary['transcript_metrics']
        self.assertEqual(len(transcript_metrics), 2)
        
        # Check SIRV101 metrics
        self.assertEqual(transcript_metrics.loc['SIRV101', 'expected_count'], 18)
        self.assertEqual(transcript_metrics.loc['SIRV101', 'observed_count'], 14)
        self.assertEqual(transcript_metrics.loc['SIRV101', 'detected'], 1.0)
        
        # Check SIRV201 metrics
        self.assertEqual(transcript_metrics.loc['SIRV201', 'expected_count'], 9)
        self.assertEqual(transcript_metrics.loc['SIRV201', 'observed_count'], 4)
        self.assertEqual(transcript_metrics.loc['SIRV201', 'detected'], 0.5)
        
        # Check cell metrics
        self.assertIn('cell_metrics', summary)
        cell_metrics = summary['cell_metrics']
        self.assertEqual(len(cell_metrics), 2)
        
        # Check CELL1 metrics
        self.assertEqual(cell_metrics.loc['CELL1', 'expected_count'], 15)
        self.assertEqual(cell_metrics.loc['CELL1', 'observed_count'], 12)
        self.assertEqual(cell_metrics.loc['CELL1', 'detected'], 1.0)
        
        # Check CELL2 metrics
        self.assertEqual(cell_metrics.loc['CELL2', 'expected_count'], 12)
        self.assertEqual(cell_metrics.loc['CELL2', 'observed_count'], 6)
        self.assertEqual(cell_metrics.loc['CELL2', 'detected'], 0.5)
    
    @patch('sirv_pipeline.evaluation.plt.figure')
    @patch('sirv_pipeline.evaluation.plt.scatter')
    @patch('sirv_pipeline.evaluation.plt.plot')
    @patch('sirv_pipeline.evaluation.plt.savefig')
    @patch('sirv_pipeline.evaluation.plt.close')
    def test_generate_plots(self, mock_close, mock_savefig, mock_plot, mock_scatter, mock_figure):
        """Test generating evaluation plots"""
        # Create comparison DataFrame
        comparison_data = {
            'barcode': ['CELL1', 'CELL1', 'CELL2', 'CELL2'],
            'sirv_transcript': ['SIRV101', 'SIRV201', 'SIRV101', 'SIRV201'],
            'expected_count': [10, 5, 8, 4],
            'observed_count': [8, 4, 6, 0],
            'detected': [True, True, True, False],
            'detection_rate': [0.8, 0.8, 0.75, 0.0]
        }
        comparison = pd.DataFrame(comparison_data)
        
        # Create plot directory
        plot_dir = os.path.join(self.temp_dir.name, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Test function
        _generate_plots(comparison, plot_dir)
        
        # Check results
        self.assertEqual(mock_figure.call_count, 3)  # Three plots
        self.assertEqual(mock_savefig.call_count, 3)  # Three plot files
        self.assertEqual(mock_close.call_count, 3)   # Close three plots
    
    def test_compare_with_flames(self):
        """Test comparing expected vs observed SIRV counts"""
        # Create output directory
        output_file = os.path.join(self.temp_dir.name, "comparison.csv")
        plot_dir = os.path.join(self.temp_dir.name, "plots")
        
        # Test function
        with patch('sirv_pipeline.evaluation._generate_plots'):
            comparison = compare_with_flames(
                expected_file=self.expected_file,
                flames_output=self.flames_file,
                output_file=output_file,
                plot_dir=plot_dir
            )
        
        # Check results
        self.assertEqual(len(comparison), 4)
        self.assertTrue(os.path.exists(output_file))
        
        # Check output file
        df = pd.read_csv(output_file)
        self.assertEqual(len(df), 4)
        self.assertEqual(list(df.columns), ['barcode', 'sirv_transcript', 'expected_count', 
                                        'observed_count', 'detected', 'detection_rate'])
        
        # Check individual entries
        # CELL1, SIRV101: expected 10, observed 8
        cell1_sirv101 = df[(df['barcode'] == 'CELL1') & (df['sirv_transcript'] == 'SIRV101')].iloc[0]
        self.assertEqual(cell1_sirv101['expected_count'], 10)
        self.assertEqual(cell1_sirv101['observed_count'], 8)
        self.assertEqual(cell1_sirv101['detected'], True)
        self.assertEqual(cell1_sirv101['detection_rate'], 0.8)
        
        # CELL2, SIRV201: expected 4, observed 0 (not detected)
        cell2_sirv201 = df[(df['barcode'] == 'CELL2') & (df['sirv_transcript'] == 'SIRV201')].iloc[0]
        self.assertEqual(cell2_sirv201['expected_count'], 4)
        self.assertEqual(cell2_sirv201['observed_count'], 0)
        self.assertEqual(cell2_sirv201['detected'], False)
        self.assertEqual(cell2_sirv201['detection_rate'], 0)
    
    @patch('sirv_pipeline.evaluation.jinja2')
    def test_generate_report(self, mock_jinja2):
        """Test generating HTML report"""
        # Mock jinja2.Template
        mock_template = MagicMock()
        mock_template.render.return_value = "<html>Test Report</html>"
        mock_jinja2.Template.return_value = mock_template
        
        # Create comparison file
        comparison_file = os.path.join(self.temp_dir.name, "comparison.csv")
        comparison_data = {
            'barcode': ['CELL1', 'CELL1', 'CELL2', 'CELL2'],
            'sirv_transcript': ['SIRV101', 'SIRV201', 'SIRV101', 'SIRV201'],
            'expected_count': [10, 5, 8, 4],
            'observed_count': [8, 4, 6, 0],
            'detected': [True, True, True, False],
            'detection_rate': [0.8, 0.8, 0.75, 0.0]
        }
        pd.DataFrame(comparison_data).to_csv(comparison_file, index=False)
        
        # Create output HTML file
        output_html = os.path.join(self.temp_dir.name, "report.html")
        
        # Test function
        result = generate_report(
            comparison_file=comparison_file,
            output_html=output_html
        )
        
        # Check results
        self.assertEqual(result, output_html)
        self.assertTrue(os.path.exists(output_html))
        
        # Check file content
        with open(output_html, 'r') as f:
            content = f.read()
            self.assertEqual(content, "<html>Test Report</html>")
        
        # Check jinja2.Template call
        mock_jinja2.Template.assert_called_once()
        mock_template.render.assert_called_once()


if __name__ == '__main__':
    unittest.main()