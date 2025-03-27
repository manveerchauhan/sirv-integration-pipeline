"""
Unit tests for the utilities module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import logging
import tempfile
import subprocess

# Add parent directory to path for importing the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sirv_pipeline.utils import (
    setup_logger,
    check_external_tools,
    check_file_exists,
    check_output_writable,
    get_file_size,
    human_readable_size,
    validate_insertion_rate,
    check_dependencies
)


class TestUtils(unittest.TestCase):
    """Test case for utilities module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test file
        self.test_file = os.path.join(self.temp_dir.name, "test.txt")
        with open(self.test_file, 'w') as f:
            f.write("test content")
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.temp_dir.cleanup()
    
    def test_setup_logger(self):
        """Test setting up logger"""
        # Test with console only
        logger = setup_logger(name="test_console", console=True, log_file=None)
        self.assertEqual(logger.name, "test_console")
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
        
        # Test with log file
        log_file = os.path.join(self.temp_dir.name, "test.log")
        logger = setup_logger(name="test_file", console=False, log_file=log_file)
        self.assertEqual(logger.name, "test_file")
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.FileHandler)
        
        # Test with both console and log file
        logger = setup_logger(name="test_both", console=True, log_file=log_file)
        self.assertEqual(logger.name, "test_both")
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 2)
        
        # Test with custom level
        logger = setup_logger(name="test_level", level=logging.DEBUG)
        self.assertEqual(logger.level, logging.DEBUG)
    
    @patch('sirv_pipeline.utils.subprocess.run')
    def test_check_external_tools(self, mock_run):
        """Test checking external tools"""
        # Mock subprocess.run to simulate tools available
        mock_run.return_value.returncode = 0
        
        # Test function
        tools = check_external_tools()
        
        # Check results
        self.assertIn('minimap2', tools)
        self.assertIn('samtools', tools)
        self.assertTrue(tools['minimap2'])
        self.assertTrue(tools['samtools'])
        
        # Mock subprocess.run to simulate tools not available
        mock_run.side_effect = subprocess.CalledProcessError(1, 'which')
        
        # Test function
        tools = check_external_tools()
        
        # Check results
        self.assertFalse(tools['minimap2'])
        self.assertFalse(tools['samtools'])
    
    def test_check_file_exists(self):
        """Test checking if a file exists"""
        # Test with existing file
        self.assertTrue(check_file_exists(self.test_file))
        
        # Test with non-existent file
        non_existent = os.path.join(self.temp_dir.name, "non_existent.txt")
        self.assertFalse(check_file_exists(non_existent))
    
    def test_check_output_writable(self):
        """Test checking if an output path is writable"""
        # Test with existing directory
        output_file = os.path.join(self.temp_dir.name, "output.txt")
        self.assertTrue(check_output_writable(output_file))
        
        # Test with non-existent directory that can be created
        new_dir = os.path.join(self.temp_dir.name, "new_dir")
        output_file = os.path.join(new_dir, "output.txt")
        self.assertTrue(check_output_writable(output_file))
        self.assertTrue(os.path.exists(new_dir))
        
        # Test with non-writable directory
        with patch('os.access', return_value=False):
            self.assertFalse(check_output_writable(output_file))
        
        # Test with directory creation failure
        with patch('os.makedirs', side_effect=PermissionError):
            non_existent = os.path.join("/non_existent_dir", "output.txt")
            self.assertFalse(check_output_writable(non_existent))
    
    def test_get_file_size(self):
        """Test getting file size"""
        # Test with existing file
        size = get_file_size(self.test_file)
        self.assertEqual(size, 12)  # "test content" is 12 bytes
        
        # Test with non-existent file
        non_existent = os.path.join(self.temp_dir.name, "non_existent.txt")
        size = get_file_size(non_existent)
        self.assertEqual(size, 0)
    
    def test_human_readable_size(self):
        """Test converting bytes to human-readable size"""
        # Test with various sizes
        self.assertEqual(human_readable_size(0), "0 B")
        self.assertEqual(human_readable_size(100), "100.00 B")
        self.assertEqual(human_readable_size(1024), "1.00 KB")
        self.assertEqual(human_readable_size(1024 * 1024), "1.00 MB")
        self.assertEqual(human_readable_size(1024 * 1024 * 1024), "1.00 GB")
        self.assertEqual(human_readable_size(1024 * 1024 * 1024 * 1024), "1.00 TB")
    
    def test_validate_insertion_rate(self):
        """Test validating insertion rate"""
        # Test with valid rates
        self.assertEqual(validate_insertion_rate(0.01), 0.01)
        self.assertEqual(validate_insertion_rate(0.1), 0.1)
        self.assertEqual(validate_insertion_rate(0.5), 0.5)
        
        # Test with invalid rates
        with self.assertRaises(ValueError):
            validate_insertion_rate(0)
        
        with self.assertRaises(ValueError):
            validate_insertion_rate(-0.1)
        
        with self.assertRaises(ValueError):
            validate_insertion_rate(0.6)
    
    @patch('sirv_pipeline.utils.check_external_tools')
    def test_check_dependencies(self, mock_check_tools):
        """Test checking dependencies"""
        # Mock check_external_tools to return all tools available
        mock_check_tools.return_value = {
            'minimap2': True,
            'samtools': True
        }
        
        # Test function with all dependencies available
        with patch('builtins.__import__', return_value=None):
            self.assertTrue(check_dependencies())
        
        # Test function with missing external tools
        mock_check_tools.return_value = {
            'minimap2': False,
            'samtools': True
        }
        
        with patch('builtins.print'):
            self.assertFalse(check_dependencies())
        
        # Test function with missing Python dependencies
        mock_check_tools.return_value = {
            'minimap2': True,
            'samtools': True
        }
        
        with patch('builtins.__import__', side_effect=ImportError):
            with patch('builtins.print'):
                self.assertFalse(check_dependencies())


if __name__ == '__main__':
    unittest.main()