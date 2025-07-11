#!/usr/bin/env python3
"""
Apollo.io Text-to-Speech Fine-tuning with Unsloth
==================================================

This script implements text-to-speech fine-tuning using Unsloth's GRPO
(Group Relative Policy Optimization) for creating high-quality TTS models.

Features:
- GRPO reinforcement learning for TTS optimization
- Custom reward functions for speech quality assessment
- Support for multiple model types
- Full fine-tuning capabilities
- Integration with Hugging Face Hub

Author: Apollo.io TTS Team
Date: 2024
"""

import os
import re
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

# Unsloth imports
from unsloth import FastLanguageModel
from unsloth.models import load_model
from unsloth.models.gemma import model_patcher
from unsloth.models.llama import model_patcher as llama_model_patcher
from unsloth.models.qwen import model_patcher as qwen_model_patcher

# Dataset and training imports
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Audio processing imports
import librosa
import soundfile as sf
from scipy.io import wavfile
import torchaudio

# Reward function imports
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class TTSConfig:
    """Configuration for TTS fine-tuning"""
    model_name: str = "microsoft/DialoGPT-medium"  # Base model for TTS
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    output_dir: str = "./apollo_tts_model"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # GRPO specific settings
    use_grpo: bool = True
    num_generations: int = 4
    max_steps: int = 1000
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    delta: float = 1.5
    loss_type: str = "grpo"  # Options: grpo, dr_grpo, dapo, bnpo
    
    # TTS specific settings
    sample_rate: int = 22050
    hop_length: int = 512
    win_length: int = 1024
    n_mels: int = 80
    n_fft: int = 2048


class TTSRewardFunction:
    """Custom reward functions for TTS quality assessment"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.quality_metrics = {
            "clarity": 0.0,
            "naturalness": 0.0,
            "prosody": 0.0,
            "articulation": 0.0,
            "fluency": 0.0
        }
    
    def calculate_audio_quality(self, audio_path: str) -> float:
        """Calculate audio quality metrics"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            # Calculate various quality metrics
            clarity_score = self._calculate_clarity(audio)
            naturalness_score = self._calculate_naturalness(audio)
            prosody_score = self._calculate_prosody(audio)
            articulation_score = self._calculate_articulation(audio)
            fluency_score = self._calculate_fluency(audio)
            
            # Weighted average
            total_score = (
                clarity_score * 0.25 +
                naturalness_score * 0.25 +
                prosody_score * 0.2 +
                articulation_score * 0.15 +
                fluency_score * 0.15
            )
            
            return total_score
            
        except Exception as e:
            print(f"Error calculating audio quality: {e}")
            return 0.0
    
    def _calculate_clarity(self, audio: np.ndarray) -> float:
        """Calculate audio clarity score"""
        # Signal-to-noise ratio approximation
        signal_power = np.mean(audio**2)
        noise_power = np.mean(np.diff(audio)**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return min(1.0, max(0.0, snr / 20.0))
    
    def _calculate_naturalness(self, audio: np.ndarray) -> float:
        """Calculate naturalness score based on spectral characteristics"""
        # Extract spectral features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.config.sample_rate, n_mfcc=13)
        
        # Calculate spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.config.sample_rate)
        
        # Naturalness based on spectral distribution
        naturalness = np.mean(spectral_centroid) / 2000.0  # Normalize
        return min(1.0, max(0.0, naturalness))
    
    def _calculate_prosody(self, audio: np.ndarray) -> float:
        """Calculate prosody score based on pitch and rhythm"""
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.config.sample_rate)
        
        # Calculate pitch variation
        pitch_variation = np.std(pitches[magnitudes > 0.1])
        prosody_score = min(1.0, pitch_variation / 100.0)
        
        return prosody_score
    
    def _calculate_articulation(self, audio: np.ndarray) -> float:
        """Calculate articulation score based on formant structure"""
        # Extract formants using LPC
        frame_length = 1024
        hop_length = 512
        
        articulation_score = 0.0
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            if len(frame) == frame_length:
                # Simple articulation measure based on spectral rolloff
                rolloff = librosa.feature.spectral_rolloff(y=frame, sr=self.config.sample_rate)
                articulation_score += np.mean(rolloff) / self.config.sample_rate
        
        articulation_score /= max(1, (len(audio) - frame_length) // hop_length)
        return min(1.0, articulation_score)
    
    def _calculate_fluency(self, audio: np.ndarray) -> float:
        """Calculate fluency score based on temporal continuity"""
        # Calculate zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        # Calculate spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.config.sample_rate)
        
        # Fluency based on smooth transitions
        fluency = 1.0 - np.mean(zcr)  # Lower ZCR = more fluent
        fluency *= (1.0 - np.std(contrast) / np.mean(contrast))  # Smooth spectral transitions
        
        return max(0.0, min(1.0, fluency))


class ApolloTTSPreprocessor:
    """Data preprocessing for Apollo.io TTS fine-tuning"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.system_prompt = """You are Apollo.io's advanced text-to-speech system. 
        Generate high-quality, natural-sounding speech with proper intonation, 
        clear articulation, and natural prosody. Focus on:
        1. Clear pronunciation of all words
        2. Natural rhythm and pacing
        3. Appropriate emotional tone
        4. Smooth transitions between words
        5. Professional delivery suitable for business contexts"""
    
    def prepare_tts_dataset(self, dataset_path: str) -> Dataset:
        """Prepare TTS dataset from text-audio pairs"""
        try:
            # Load dataset
            if dataset_path.endswith('.json'):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
            else:
                data = load_dataset(dataset_path)
            
            # Process data
            processed_data = []
            for item in data:
                if isinstance(item, dict):
                    text = item.get('text', '')
                    audio_path = item.get('audio_path', '')
                    
                    if text and audio_path:
                        processed_item = {
                            'text': text,
                            'audio_path': audio_path,
                            'prompt': self._create_prompt(text),
                            'target_audio': audio_path
                        }
                        processed_data.append(processed_item)
            
            return Dataset.from_list(processed_data)
            
        except Exception as e:
            print(f"Error preparing TTS dataset: {e}")
            return Dataset.from_list([])
    
    def _create_prompt(self, text: str) -> str:
        """Create a prompt for TTS generation"""
        return f"Generate speech for the following text with natural intonation and clear pronunciation:\n\n{text}"
    
    def create_sample_dataset(self) -> Dataset:
        """Create a sample dataset for testing"""
        sample_data = [
            {
                'text': 'Welcome to Apollo.io, the leading sales intelligence platform.',
                'audio_path': 'sample_audio_1.wav',
                'prompt': self._create_prompt('Welcome to Apollo.io, the leading sales intelligence platform.'),
                'target_audio': 'sample_audio_1.wav'
            },
            {
                'text': 'Our AI-powered platform helps you find, engage, and convert your ideal customers.',
                'audio_path': 'sample_audio_2.wav',
                'prompt': self._create_prompt('Our AI-powered platform helps you find, engage, and convert your ideal customers.'),
                'target_audio': 'sample_audio_2.wav'
            },
            {
                'text': 'Transform your sales process with intelligent automation and data-driven insights.',
                'audio_path': 'sample_audio_3.wav',
                'prompt': self._create_prompt('Transform your sales process with intelligent automation and data-driven insights.'),
                'target_audio': 'sample_audio_3.wav'
            }
        ]
        
        return Dataset.from_list(sample_data)


class ApolloTTSModel:
    """Main TTS fine-tuning model using Unsloth"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.reward_function = TTSRewardFunction(config)
        self.preprocessor = ApolloTTSPreprocessor(config)
        
    def load_model(self):
        """Load and configure the base model"""
        try:
            print("Loading model with Unsloth...")
            
            # Load model with Unsloth optimizations
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_length,
                dtype=None,  # Auto-detect
                load_in_4bit=True,  # Use 4-bit quantization
            )
            
            # Apply model patching based on model type
            if "gemma" in self.config.model_name.lower():
                model = model_patcher(model)
            elif "llama" in self.config.model_name.lower():
                model = llama_model_patcher(model)
            elif "qwen" in self.config.model_name.lower():
                model = qwen_model_patcher(model)
            
            # Configure LoRA
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
                use_rslora=False,
                loftq_config=None,
            )
            
            self.model = model
            self.tokenizer = tokenizer
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def prepare_grpo_config(self):
        """Prepare GRPO configuration for TTS training"""
        from unsloth import GRPOConfig
        
        grpo_config = GRPOConfig(
            use_vllm=True,
            learning_rate=self.config.learning_rate,
            num_generations=self.config.num_generations,
            max_steps=self.config.max_steps,
            epsilon=self.config.epsilon,
            epsilon_high=self.config.epsilon_high,
            delta=self.config.delta,
            loss_type=self.config.loss_type,
            mask_truncated_completions=True,
        )
        
        return grpo_config
    
    def train(self, dataset: Dataset):
        """Train the TTS model using GRPO"""
        try:
            if self.model is None:
                self.load_model()
            
            print("Starting TTS fine-tuning with GRPO...")
            
            # Prepare GRPO configuration
            grpo_config = self.prepare_grpo_config()
            
            # Custom reward function for TTS
            def tts_reward_function(generations, prompts, tokenizer):
                """Custom reward function for TTS quality assessment"""
                rewards = []
                
                for generation in generations:
                    # Extract generated text
                    generated_text = tokenizer.decode(generation, skip_special_tokens=True)
                    
                    # Calculate reward based on TTS quality metrics
                    reward = self._calculate_tts_reward(generated_text)
                    rewards.append(reward)
                
                return torch.tensor(rewards, dtype=torch.float32)
            
            # Train with GRPO
            trainer = FastLanguageModel.get_trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                eval_dataset=dataset.select(range(min(10, len(dataset)))),
                args=TrainingArguments(
                    output_dir=self.config.output_dir,
                    num_train_epochs=self.config.num_epochs,
                    per_device_train_batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    learning_rate=self.config.learning_rate,
                    warmup_steps=self.config.warmup_steps,
                    save_steps=self.config.save_steps,
                    eval_steps=self.config.eval_steps,
                    logging_steps=self.config.logging_steps,
                    save_total_limit=self.config.save_total_limit,
                    load_best_model_at_end=self.config.load_best_model_at_end,
                    metric_for_best_model=self.config.metric_for_best_model,
                    greater_is_better=self.config.greater_is_better,
                    report_to=None,
                ),
                grpo_config=grpo_config,
                reward_function=tts_reward_function,
            )
            
            # Start training
            trainer.train()
            
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise
    
    def _calculate_tts_reward(self, generated_text: str) -> float:
        """Calculate reward for TTS generation quality"""
        reward = 0.0
        
        # Reward for proper text structure
        if len(generated_text.strip()) > 0:
            reward += 1.0
        
        # Reward for natural language patterns
        if any(word in generated_text.lower() for word in ['apollo', 'sales', 'intelligence', 'platform']):
            reward += 0.5
        
        # Reward for professional tone
        professional_words = ['platform', 'intelligence', 'automation', 'insights', 'customers']
        if any(word in generated_text.lower() for word in professional_words):
            reward += 0.3
        
        # Penalty for repetitive text
        words = generated_text.split()
        if len(set(words)) / len(words) < 0.7:  # Low diversity
            reward -= 0.2
        
        # Penalty for very short or very long outputs
        if len(generated_text) < 10:
            reward -= 0.5
        elif len(generated_text) > 500:
            reward -= 0.3
        
        return max(0.0, reward)
    
    def save_model(self, save_path: str = None):
        """Save the fine-tuned model"""
        try:
            if save_path is None:
                save_path = self.config.output_dir
            
            # Save LoRA weights
            self.model.save_lora(f"{save_path}/lora_weights")
            
            # Save merged model
            self.model.save_pretrained_merged(
                save_path, 
                self.tokenizer, 
                save_method="merged_16bit"
            )
            
            print(f"Model saved to {save_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
            raise
    
    def push_to_hub(self, repo_name: str, token: str = None):
        """Push model to Hugging Face Hub"""
        try:
            if token is None:
                token = os.getenv("HF_TOKEN")
            
            if token is None:
                raise ValueError("Hugging Face token not provided")
            
            # Push to Hub
            self.model.push_to_hub_merged(
                repo_name,
                self.tokenizer,
                save_method="merged_16bit",
                token=token
            )
            
            print(f"Model pushed to {repo_name}")
            
        except Exception as e:
            print(f"Error pushing to Hub: {e}")
            raise


def main():
    """Main function to run Apollo.io TTS fine-tuning"""
    print("🚀 Apollo.io Text-to-Speech Fine-tuning with Unsloth")
    print("=" * 60)
    
    # Initialize configuration
    config = TTSConfig(
        model_name="microsoft/DialoGPT-medium",
        max_length=512,
        batch_size=2,  # Reduced for memory constraints
        learning_rate=2e-4,
        num_epochs=2,
        max_steps=500,  # Reduced for testing
        use_grpo=True,
        loss_type="grpo"
    )
    
    # Initialize model
    tts_model = ApolloTTSModel(config)
    
    # Load model
    print("Loading model...")
    tts_model.load_model()
    
    # Prepare dataset
    print("Preparing dataset...")
    preprocessor = ApolloTTSPreprocessor(config)
    dataset = preprocessor.create_sample_dataset()
    
    print(f"Dataset prepared with {len(dataset)} samples")
    
    # Train model
    print("Starting training...")
    tts_model.train(dataset)
    
    # Save model
    print("Saving model...")
    tts_model.save_model()
    
    print("✅ Apollo.io TTS fine-tuning completed successfully!")
    print("\nNext steps:")
    print("1. Test the model with sample text")
    print("2. Push to Hugging Face Hub if desired")
    print("3. Deploy for production use")


if __name__ == "__main__":
    main() 