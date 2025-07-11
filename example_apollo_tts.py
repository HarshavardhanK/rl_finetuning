#!/usr/bin/env python3
"""
Apollo.io TTS Fine-tuning Example
=================================

This example demonstrates how to use the Apollo.io TTS fine-tuning
implementation with Unsloth's GRPO for text-to-speech optimization.
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from apollo_tts_finetune import (
    TTSConfig, 
    ApolloTTSPreprocessor, 
    ApolloTTSModel
)


def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    sample_data = [
        {
            "text": "Welcome to Apollo.io, the leading sales intelligence platform.",
            "audio_path": "samples/apollo_intro.wav",
            "quality_score": 0.95
        },
        {
            "text": "Our AI-powered platform helps you find, engage, and convert your ideal customers.",
            "audio_path": "samples/apollo_features.wav", 
            "quality_score": 0.92
        },
        {
            "text": "Transform your sales process with intelligent automation and data-driven insights.",
            "audio_path": "samples/apollo_benefits.wav",
            "quality_score": 0.89
        },
        {
            "text": "Join thousands of sales professionals who trust Apollo.io for their prospecting needs.",
            "audio_path": "samples/apollo_trust.wav",
            "quality_score": 0.91
        },
        {
            "text": "Get started today and see the difference Apollo.io can make in your sales pipeline.",
            "audio_path": "samples/apollo_cta.wav",
            "quality_score": 0.88
        }
    ]
    
    # Create samples directory
    os.makedirs("samples", exist_ok=True)
    
    # Save sample dataset
    with open("samples/sample_dataset.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Sample dataset created at samples/sample_dataset.json")
    return "samples/sample_dataset.json"


def demonstrate_configuration():
    """Demonstrate different configuration options"""
    print("\n🔧 Configuration Examples")
    print("=" * 40)
    
    # Basic configuration
    basic_config = TTSConfig(
        model_name="microsoft/DialoGPT-medium",
        max_steps=100,  # Small for demo
        use_grpo=True
    )
    print("✅ Basic configuration created")
    
    # Advanced configuration
    advanced_config = TTSConfig(
        model_name="microsoft/DialoGPT-medium",
        max_length=512,
        batch_size=2,
        learning_rate=2e-4,
        num_epochs=2,
        use_grpo=True,
        loss_type="grpo",
        epsilon=0.2,
        epsilon_high=0.28,
        delta=1.5,
        sample_rate=22050,
        n_mels=80,
        max_steps=200
    )
    print("✅ Advanced configuration created")
    
    # Production configuration
    production_config = TTSConfig(
        model_name="microsoft/DialoGPT-medium",
        max_length=1024,
        batch_size=4,
        learning_rate=1e-4,
        num_epochs=5,
        use_grpo=True,
        loss_type="dr_grpo",  # Dr. GRPO for better results
        epsilon=0.15,
        epsilon_high=0.25,
        delta=1.2,
        sample_rate=22050,
        n_mels=80,
        max_steps=1000
    )
    print("✅ Production configuration created")
    
    return basic_config, advanced_config, production_config


def demonstrate_preprocessing():
    """Demonstrate data preprocessing"""
    print("\n📊 Data Preprocessing Demo")
    print("=" * 40)
    
    # Create sample dataset
    dataset_path = create_sample_dataset()
    
    # Initialize preprocessor
    config = TTSConfig()
    preprocessor = ApolloTTSPreprocessor(config)
    
    # Create sample dataset
    dataset = preprocessor.create_sample_dataset()
    print(f"✅ Sample dataset created with {len(dataset)} samples")
    
    # Show first sample
    first_sample = dataset[0]
    print(f"📝 Sample text: {first_sample['text']}")
    print(f"🎵 Audio path: {first_sample['audio_path']}")
    print(f"💬 Prompt: {first_sample['prompt'][:100]}...")
    
    return dataset


def demonstrate_reward_function():
    """Demonstrate reward function usage"""
    print("\n🎯 Reward Function Demo")
    print("=" * 40)
    
    from apollo_tts_finetune import TTSRewardFunction
    import numpy as np
    
    config = TTSConfig()
    reward_function = TTSRewardFunction(config)
    
    # Test with mock audio
    test_audio = np.random.randn(22050)  # 1 second at 22kHz
    
    # Calculate quality metrics
    clarity = reward_function._calculate_clarity(test_audio)
    naturalness = reward_function._calculate_naturalness(test_audio)
    prosody = reward_function._calculate_prosody(test_audio)
    articulation = reward_function._calculate_articulation(test_audio)
    fluency = reward_function._calculate_fluency(test_audio)
    
    print(f"🎵 Audio Quality Metrics:")
    print(f"   Clarity: {clarity:.3f}")
    print(f"   Naturalness: {naturalness:.3f}")
    print(f"   Prosody: {prosody:.3f}")
    print(f"   Articulation: {articulation:.3f}")
    print(f"   Fluency: {fluency:.3f}")
    
    # Calculate overall quality
    overall_quality = reward_function.calculate_audio_quality("mock_audio.wav")
    print(f"   Overall Quality: {overall_quality:.3f}")


def demonstrate_model_usage():
    """Demonstrate model usage without actual training"""
    print("\n🤖 Model Usage Demo")
    print("=" * 40)
    
    # Initialize configuration
    config = TTSConfig(
        model_name="microsoft/DialoGPT-medium",
        max_steps=10,  # Very small for demo
        use_grpo=True
    )
    
    # Initialize model
    tts_model = ApolloTTSModel(config)
    print("✅ TTS model initialized")
    
    # Test reward calculation
    test_texts = [
        "Welcome to Apollo.io, the leading sales intelligence platform.",
        "Our AI-powered platform helps you find, engage, and convert customers.",
        "Transform your sales process with intelligent automation.",
        "test test test test test test test test test test",  # Repetitive
        ""  # Empty
    ]
    
    print("\n🎯 Reward Function Examples:")
    for i, text in enumerate(test_texts, 1):
        reward = tts_model._calculate_tts_reward(text)
        print(f"   {i}. '{text[:50]}{'...' if len(text) > 50 else ''}' -> {reward:.3f}")
    
    return tts_model


def demonstrate_training_pipeline():
    """Demonstrate the complete training pipeline"""
    print("\n🚀 Training Pipeline Demo")
    print("=" * 40)
    
    # Step 1: Configuration
    config = TTSConfig(
        model_name="microsoft/DialoGPT-medium",
        max_steps=50,  # Very small for demo
        batch_size=1,
        use_grpo=True
    )
    print("✅ Step 1: Configuration created")
    
    # Step 2: Data preprocessing
    preprocessor = ApolloTTSPreprocessor(config)
    dataset = preprocessor.create_sample_dataset()
    print(f"✅ Step 2: Dataset prepared with {len(dataset)} samples")
    
    # Step 3: Model initialization
    tts_model = ApolloTTSModel(config)
    print("✅ Step 3: Model initialized")
    
    # Step 4: Training (simulated)
    print("✅ Step 4: Training pipeline ready")
    print("   Note: Actual training requires GPU and would take time")
    print("   To run training: tts_model.load_model(); tts_model.train(dataset)")
    
    # Step 5: Saving (simulated)
    print("✅ Step 5: Model saving ready")
    print("   To save model: tts_model.save_model('output/path')")
    
    return tts_model, dataset


def main():
    """Main demonstration function"""
    print("🎤 Apollo.io TTS Fine-tuning Demo")
    print("=" * 50)
    print("This demo shows the key components of the TTS fine-tuning system.")
    print("Note: Actual training requires GPU and real audio data.\n")
    
    try:
        # Demonstrate configuration
        basic_config, advanced_config, production_config = demonstrate_configuration()
        
        # Demonstrate preprocessing
        dataset = demonstrate_preprocessing()
        
        # Demonstrate reward function
        demonstrate_reward_function()
        
        # Demonstrate model usage
        tts_model = demonstrate_model_usage()
        
        # Demonstrate training pipeline
        model, dataset = demonstrate_training_pipeline()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Install required dependencies: pip install -r requirements_apollo_tts.txt")
        print("2. Prepare your audio dataset")
        print("3. Run the main training script: python apollo_tts_finetune.py")
        print("4. Test the model with real audio data")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("This is expected if dependencies are not installed.")
        print("Install dependencies first: pip install -r requirements_apollo_tts.txt")


if __name__ == "__main__":
    main() 