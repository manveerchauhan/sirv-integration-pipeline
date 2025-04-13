"""
Feature cache utilities for the SIRV Integration Pipeline.

This module provides a caching system for RNA sequence features
to avoid recomputing expensive calculations, significantly
speeding up the coverage modeling process.
"""

import os
import pickle
import logging
import hashlib
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

class FeatureCache:
    """
    Cache for RNA sequence features to avoid recomputation.
    
    This class maintains an in-memory cache of transcript features
    that can be saved to and loaded from disk. It significantly speeds up
    repeated analyses on the same reference sequences.
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize the feature cache.
        
        Args:
            cache_file: Path to the cache file to load/save features
        """
        self.features = {}  # Main cache dictionary: transcript_id -> features
        self.cache_file = cache_file
        self.cache_stats = {
            "hits": 0,      # Number of cache hits
            "misses": 0,    # Number of cache misses
            "adds": 0       # Number of new features added
        }
        
        # Load existing cache if available
        if cache_file and os.path.exists(cache_file):
            self._load_from_file(cache_file)
    
    def _load_from_file(self, cache_file: str) -> None:
        """
        Load features from a cache file.
        
        Args:
            cache_file: Path to the cache file
        """
        try:
            with open(cache_file, 'rb') as f:
                loaded_data = pickle.load(f)
                
                # Handle different versions of cache format
                if isinstance(loaded_data, dict):
                    if 'features' in loaded_data:
                        # New format with metadata
                        self.features = loaded_data.get('features', {})
                        # We can load additional metadata here in the future
                    else:
                        # Old format - just a dict of features
                        self.features = loaded_data
                else:
                    logger.warning(f"Invalid cache format in {cache_file}")
                    return
                
            logger.info(f"Loaded {len(self.features)} cached features from {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load feature cache from {cache_file}: {str(e)}")
            self.features = {}
    
    def get(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """
        Get features for a transcript from the cache.
        
        Args:
            transcript_id: Transcript identifier
            
        Returns:
            Dictionary of features or None if not in cache
        """
        if transcript_id in self.features:
            self.cache_stats["hits"] += 1
            return self.features[transcript_id]
        else:
            self.cache_stats["misses"] += 1
            return None
    
    def add(self, transcript_id: str, features: Dict[str, Any]) -> None:
        """
        Add features for a transcript to the cache.
        
        Args:
            transcript_id: Transcript identifier
            features: Dictionary of calculated features
        """
        self.features[transcript_id] = features
        self.cache_stats["adds"] += 1
    
    def add_sequence_features(self, transcript_id: str, sequence: str) -> Dict[str, Any]:
        """
        Calculate and add features for a sequence.
        
        Args:
            transcript_id: Transcript identifier
            sequence: Nucleotide sequence
            
        Returns:
            Dictionary of calculated features
        """
        # Check if already in cache first
        if transcript_id in self.features:
            self.cache_stats["hits"] += 1
            return self.features[transcript_id]
        
        # Calculate features
        features = self._calculate_sequence_features(sequence)
        
        # Add to cache
        self.features[transcript_id] = features
        self.cache_stats["adds"] += 1
        
        return features
    
    def _calculate_sequence_features(self, sequence: str) -> Dict[str, Any]:
        """
        Calculate RNA features for a sequence.
        
        Args:
            sequence: Nucleotide sequence
            
        Returns:
            Dictionary of calculated features
        """
        sequence = sequence.upper()  # Normalize sequence
        length = len(sequence)
        
        if length == 0:
            return {
                'length': 0,
                'gc_content': 0.0,
                'five_prime_gc': 0.0,
                'three_prime_gc': 0.0,
                'complexity': 0.0,
                'hairpin_potential': 0.0,
                'g_quadruplex_potential': 0.0,
                'homopolymer_ratio': 0.0,
                'gc_skew': 0.0
            }
        
        # Basic composition
        g_count = sequence.count('G')
        c_count = sequence.count('C')
        a_count = sequence.count('A')
        t_count = sequence.count('T')  # Also counts U in RNA
        
        gc_count = g_count + c_count
        gc_content = gc_count / length
        
        # GC skew
        gc_skew = (g_count - c_count) / (g_count + c_count) if (g_count + c_count) > 0 else 0
        
        # Terminal regions
        five_prime_region = sequence[:min(50, length)]
        three_prime_region = sequence[max(0, length - 50):]
        
        five_prime_gc = self._calculate_gc(five_prime_region)
        three_prime_gc = self._calculate_gc(three_prime_region)
        
        # Sequence complexity (k-mer diversity)
        k = min(4, length // 2) if length > 4 else 1
        complexity = self._calculate_complexity(sequence, k)
        
        # RNA structure potential
        hairpin_potential = self._estimate_hairpin_potential(sequence)
        g_quadruplex_potential = self._estimate_g_quadruplex_potential(sequence)
        
        # Homopolymer content
        homopolymer_ratio = self._calculate_homopolymer_ratio(sequence)
        
        # Return all features
        return {
            'length': length,
            'gc_content': gc_content,
            'five_prime_gc': five_prime_gc,
            'three_prime_gc': three_prime_gc,
            'complexity': complexity,
            'hairpin_potential': hairpin_potential,
            'g_quadruplex_potential': g_quadruplex_potential,
            'homopolymer_ratio': homopolymer_ratio,
            'gc_skew': gc_skew
        }
    
    def _calculate_gc(self, sequence: str) -> float:
        """Calculate GC content of a sequence."""
        if not sequence:
            return 0.0
        return (sequence.count('G') + sequence.count('C')) / len(sequence)
    
    def _calculate_complexity(self, sequence: str, k: int) -> float:
        """Calculate sequence complexity based on k-mer diversity."""
        if len(sequence) < k:
            return 0.0
            
        kmers = set()
        for i in range(len(sequence) - k + 1):
            kmers.add(sequence[i:i+k])
            
        # Normalize by maximum possible k-mers
        max_kmers = min(4**k, len(sequence) - k + 1)
        return len(kmers) / max_kmers if max_kmers > 0 else 0.0
    
    def _estimate_hairpin_potential(self, sequence: str) -> float:
        """Estimate hairpin formation potential."""
        if len(sequence) < 30:
            return 0.0
            
        pairs = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        max_complementary = 0
        
        for i in range(len(sequence) - 30):
            complementary = 0
            for j in range(10):
                if i + j < len(sequence) and i + 20 + j < len(sequence):
                    if pairs.get(sequence[i + j]) == sequence[i + 20 + j]:
                        complementary += 1
            max_complementary = max(max_complementary, complementary)
            
        return max_complementary / 10
    
    def _estimate_g_quadruplex_potential(self, sequence: str) -> float:
        """Estimate G-quadruplex formation potential."""
        if len(sequence) < 15:
            return 0.0
            
        g_runs = 0
        for i in range(len(sequence) - 15):
            window = sequence[i:i+15]
            g_count = window.count('G')
            if g_count > 10:  # High G content in small window
                g_runs += 1
                
        return g_runs / (len(sequence) - 14) if len(sequence) > 14 else 0.0
    
    def _calculate_homopolymer_ratio(self, sequence: str) -> float:
        """Calculate the longest homopolymer run ratio."""
        if len(sequence) == 0:
            return 0.0
            
        max_run = 0
        for base in 'ACGT':
            current_run = 0
            for char in sequence:
                if char == base:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
                    
        return max_run / len(sequence)
    
    def save(self) -> bool:
        """
        Save the cache to disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.cache_file:
            logger.warning("No cache file specified, cannot save cache")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
            
            # Save cache with metadata
            cache_data = {
                'features': self.features,
                'stats': self.cache_stats,
                'version': '1.0'
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"Saved {len(self.features)} features to {self.cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save feature cache to {self.cache_file}: {str(e)}")
            return False
    
    def clear(self) -> None:
        """Clear the cache."""
        self.features = {}
        logger.info("Feature cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            **self.cache_stats,
            "size": len(self.features)
        }
    
    def __len__(self) -> int:
        """Get the number of cached features."""
        return len(self.features)
    
    def __contains__(self, transcript_id: str) -> bool:
        """Check if a transcript is in the cache."""
        return transcript_id in self.features 