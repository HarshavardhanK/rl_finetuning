# LLM Fine-tuning with GRPO and Unsloth

This repository contains implementations for fine-tuning Large Language Models (LLMs) using Group Relative Policy Optimization (GRPO) with the Unsloth framework. The project focuses on mathematical reasoning tasks using the GSM8K dataset.

## 🚀 Features

- **GRPO Training**: Implementation of Group Relative Policy Optimization for RLHF
- **Mathematical Reasoning**: Fine-tuned models for solving math problems
- **Multiple Model Support**: Llama 2.1, Llama 3.1, and Llama 3.2 variants
- **LoRA Integration**: Efficient parameter-efficient fine-tuning
- **Distributed Training**: Support for multi-GPU training
- **Unsloth Framework**: Optimized training with memory efficiency

## 📋 Requirements

### System Requirements
- CUDA-compatible GPU(s)
- Python 3.10+
- At least 16GB GPU memory (for 8B models)
- 8GB GPU memory (for 3B models)

### Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `unsloth` - Optimized training framework
- `bitsandbytes` - Quantization support
- `accelerate` - Distributed training
- `xformers` - Memory-efficient attention
- `peft` - Parameter-efficient fine-tuning
- `trl` - Transformer Reinforcement Learning
- `triton` - GPU kernel optimization
- `datasets` - Hugging Face datasets
- `transformers` - Hugging Face transformers

## 🏗️ Project Structure

```
├── reasoning/                    # Main training scripts
│   ├── grpo_training.py         # Basic GRPO training for Llama 3.1 8B
│   ├── advanced_grpo.py         # Advanced GRPO with custom formatting
│   ├── advanced_grpo_training.py # Enhanced training with multiple reward functions
│   ├── notebook.py              # Jupyter notebook conversion
│   ├── outputs/                 # Training outputs and checkpoints
│   └── grpo_trainer_lora_model/ # Trained LoRA adapters
├── download.sh                  # Model download script
├── finetune.sh                  # Distributed training script
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd finetune

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Base Model

```bash
# Set your Hugging Face token
export HF_TOKEN="your_token_here"

# Download the base model
bash download.sh
```

### 3. Run Training

#### Basic GRPO Training
```bash
cd reasoning
python grpo_training.py
```

#### Advanced GRPO Training
```bash
cd reasoning
python advanced_grpo_training.py
```

#### Distributed Training
```bash
# Set GPU devices
export CUDA_VISIBLE_DEVICES=1,5

# Run distributed training
bash ../finetune.sh
```

## 📊 Training Configuration

### Model Variants

| Model | Parameters | Memory Usage | Use Case |
|-------|------------|--------------|----------|
| Llama 3.1 8B | 8B | ~16GB | High-quality reasoning |
| Llama 3.2 3B | 3B | ~8GB | Fast inference |
| Llama 2.1 7B | 7B | ~14GB | Balanced performance |

### Training Parameters

- **LoRA Rank**: 32-64 (higher = smarter but slower)
- **Max Sequence Length**: 1024-2048 tokens
- **Learning Rate**: 5e-6
- **Batch Size**: 1 (per device)
- **Gradient Accumulation**: 1-4 steps
- **Training Steps**: 250-1000

### Reward Functions

The training uses multiple reward functions:

1. **Correctness Reward**: 2.0 points for correct answers
2. **Format Reward**: 0.5 points for proper XML formatting
3. **Integer Reward**: 0.5 points for numeric answers
4. **XML Count Reward**: Partial credit for formatting elements

## 🧮 Mathematical Reasoning

The project focuses on the GSM8K dataset for mathematical reasoning. Models are trained to:

- Follow structured reasoning format
- Provide step-by-step solutions
- Extract final numerical answers
- Maintain consistent formatting

### Output Format

```
<reasoning>
Step-by-step solution process...
</reasoning>
<answer>
Final numerical answer
</answer>
```

## 🔧 Customization

### Adding New Reward Functions

```python
def custom_reward_func(prompts, completions, answer, **kwargs):
    # Your reward logic here
    return [score for score in scores]
```

### Modifying Training Configuration

Edit the `GRPOConfig` parameters in the training scripts:

```python
training_args = GRPOConfig(
    learning_rate=5e-6,
    max_steps=250,
    per_device_train_batch_size=1,
    # Add your custom parameters
)
```

### Using Different Models

Change the model name in the training scripts:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",  # Change this
    # ... other parameters
)
```

## 📈 Monitoring Training

Training progress can be monitored through:

- Console output showing reward scores
- Log files in the `outputs/` directory
- Model checkpoints saved every 250 steps

## 🚀 Inference

After training, you can use the fine-tuned model:

```python
from unsloth import FastLanguageModel

# Load the trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    "path/to/trained/model",
    max_seq_length=1024,
    load_in_4bit=True,
)

# Generate responses
inputs = tokenizer("Solve: 2x + 5 = 13", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Unsloth](https://unsloth.ai/) for the optimized training framework
- [Hugging Face](https://huggingface.co/) for the transformers library
- [OpenAI](https://openai.com/) for the GSM8K dataset
- [Meta](https://ai.meta.com/) for the Llama models

## 📞 Support

For questions and support:
- Join the [Unsloth Discord](https://discord.gg/unsloth)
- Check the [Unsloth Documentation](https://docs.unsloth.ai/)
- Open an issue in this repository

---

**Note**: This project requires appropriate model licenses and API access. Make sure you have the necessary permissions to use the base models. 