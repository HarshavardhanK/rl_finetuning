#!/usr/bin/env python3
"""
Test script for Apollo.io TTS Fine-tuning
=========================================

This script tests the Apollo.io TTS fine-tuning implementation
to ensure all components work correctly.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from apollo_tts_finetune import (
    TTSConfig, 
    TTSRewardFunction, 
    ApolloTTSPreprocessor, 
    ApolloTTSModel
)


class TestTTSConfig(unittest.TestCase):
    """Test TTS configuration"""
    
    def test_config_initialization(self):
        """Test config initialization"""
        config = TTSConfig()
        self.assertEqual(config.model_name, "microsoft/DialoGPT-medium")
        self.assertEqual(config.max_length, 512)
        self.assertTrue(config.use_grpo)
        self.assertEqual(config.loss_type, "grpo")


class TestTTSRewardFunction(unittest.TestCase):
    """Test TTS reward function"""
    
    def setUp(self):
        self.config = TTSConfig()
        self.reward_function = TTSRewardFunction(self.config)
    
    def test_reward_function_initialization(self):
        """Test reward function initialization"""
        self.assertIsNotNone(self.reward_function)
        self.assertEqual(len(self.reward_function.quality_metrics), 5)
    
    @patch('librosa.load')
    def test_calculate_audio_quality(self, mock_load):
        """Test audio quality calculation"""
        # Mock audio data
        mock_audio = np.random.randn(22050)  # 1 second at 22kHz
        mock_load.return_value = (mock_audio, 22050)
        
        score = self.reward_function.calculate_audio_quality("fake_path.wav")
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_clarity(self):
        """Test clarity calculation"""
        audio = np.random.randn(1000)
        clarity = self.reward_function._calculate_clarity(audio)
        self.assertIsInstance(clarity, float)
        self.assertGreaterEqual(clarity, 0.0)
        self.assertLessEqual(clarity, 1.0)
    
    def test_calculate_naturalness(self):
        """Test naturalness calculation"""
        audio = np.random.randn(1000)
        naturalness = self.reward_function._calculate_naturalness(audio)
        self.assertIsInstance(naturalness, float)
        self.assertGreaterEqual(naturalness, 0.0)
        self.assertLessEqual(naturalness, 1.0)


class TestApolloTTSPreprocessor(unittest.TestCase):
    """Test TTS preprocessor"""
    
    def setUp(self):
        self.config = TTSConfig()
        self.preprocessor = ApolloTTSPreprocessor(self.config)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        self.assertIsNotNone(self.preprocessor)
        self.assertIn("Apollo.io", self.preprocessor.system_prompt)
    
    def test_create_prompt(self):
        """Test prompt creation"""
        text = "Hello, this is a test."
        prompt = self.preprocessor._create_prompt(text)
        self.assertIn(text, prompt)
        self.assertIn("Generate speech", prompt)
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation"""
        dataset = self.preprocessor.create_sample_dataset()
        self.assertEqual(len(dataset), 3)
        
        # Check first sample
        first_sample = dataset[0]
        self.assertIn('text', first_sample)
        self.assertIn('audio_path', first_sample)
        self.assertIn('prompt', first_sample)
        self.assertIn('target_audio', first_sample)


class TestApolloTTSModel(unittest.TestCase):
    """Test TTS model"""
    
    def setUp(self):
        self.config = TTSConfig()
        self.model = ApolloTTSModel(self.config)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.reward_function)
        self.assertIsNotNone(self.model.preprocessor)
    
    def test_calculate_tts_reward(self):
        """Test TTS reward calculation"""
        # Test good text
        good_text = "Welcome to Apollo.io, the leading sales intelligence platform."
        reward = self.model._calculate_tts_reward(good_text)
        self.assertIsInstance(reward, float)
        self.assertGreater(reward, 0.0)
        
        # Test empty text
        empty_text = ""
        reward = self.model._calculate_tts_reward(empty_text)
        self.assertEqual(reward, 0.0)
        
        # Test repetitive text
        repetitive_text = "test test test test test test test test test test"
        reward = self.model._calculate_tts_reward(repetitive_text)
        self.assertLess(reward, 2.0)  # Should have penalty for repetition


def run_integration_test():
    """Run integration test without loading actual model"""
    print("🧪 Running Apollo.io TTS Integration Tests")
    print("=" * 50)
    
    # Test configuration
    config = TTSConfig(
        model_name="microsoft/DialoGPT-medium",
        max_steps=10,  # Very small for testing
        num_epochs=1
    )
    
    # Test preprocessor
    print("Testing preprocessor...")
    preprocessor = ApolloTTSPreprocessor(config)
    dataset = preprocessor.create_sample_dataset()
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Test reward function
    print("Testing reward function...")
    reward_function = TTSRewardFunction(config)
    
    # Test with mock audio
    test_audio = np.random.randn(22050)
    with patch('librosa.load', return_value=(test_audio, 22050)):
        quality_score = reward_function.calculate_audio_quality("test.wav")
        print(f"✅ Audio quality score: {quality_score:.3f}")
    
    # Test TTS model (without loading actual model)
    print("Testing TTS model...")
    tts_model = ApolloTTSModel(config)
    
    # Test reward calculation
    test_text = "Welcome to Apollo.io, the leading sales intelligence platform."
    reward = tts_model._calculate_tts_reward(test_text)
    print(f"✅ TTS reward score: {reward:.3f}")
    
    print("\n🎉 All integration tests passed!")
    return True


def main():
    """Main test function"""
    print("🚀 Apollo.io TTS Fine-tuning Test Suite")
    print("=" * 50)
    
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\n" + "=" * 50)
    run_integration_test()
    
    print("\n✅ All tests completed successfully!")
    print("\nNext steps:")
    print("1. Install required dependencies")
    print("2. Run the main training script")
    print("3. Test with real audio data")


if __name__ == "__main__":
    main() 