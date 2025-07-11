#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama 3.1 8B GRPO Training Script

This script implements GRPO (Group Relative Policy Optimization) training for
Llama 3.1 8B Instruct model on GSM8K dataset for mathematical reasoning.

Original notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb
"""

import os
import re
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams


def setup_model_and_tokenizer():
    """Setup the model and tokenizer with LoRA configuration."""
    print("Setting up model and tokenizer...")
    
    max_seq_length = 1024  # Can increase for longer reasoning traces
    lora_rank = 32  # Larger rank = smarter, but slower
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,  # Reduce if out of memory
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )
    
    return model, tokenizer


def setup_data():
    """Setup the GSM8K dataset and data processing functions."""
    print("Setting up dataset...")
    
    # System prompt for reasoning format
    SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
    
    def extract_xml_answer(text: str) -> str:
        """Extract answer from XML format."""
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    def extract_hash_answer(text: str) -> str | None:
        """Extract answer from hash format."""
        if "####" not in text:
            return None
        return text.split("####")[1].strip()
    
    def get_gsm8k_questions(split="train") -> Dataset:
        """Load and prepare GSM8K dataset."""
        data = load_dataset('openai/gsm8k', 'main')[split]
        data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_hash_answer(x['answer'])
        })
        return data
    
    dataset = get_gsm8k_questions()
    return dataset, SYSTEM_PROMPT, extract_xml_answer


def setup_reward_functions(extract_xml_answer):
    """Setup reward functions for GRPO training."""
    print("Setting up reward functions...")
    
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        """Reward function for correct answers."""
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [extract_xml_answer(r) for r in responses]
        print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", 
              f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    
    def int_reward_func(completions, **kwargs) -> list[float]:
        """Reward function for integer answers."""
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
    
    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]
    
    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]
    
    def count_xml(text) -> float:
        """Count XML formatting elements."""
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1])*0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
        return count
    
    def xmlcount_reward_func(completions, **kwargs) -> list[float]:
        """Reward function based on XML formatting."""
        contents = [completion[0]["content"] for completion in completions]
        return [count_xml(c) for c in contents]
    
    return [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]


def setup_training_config(max_seq_length):
    """Setup GRPO training configuration."""
    print("Setting up training configuration...")
    
    max_prompt_length = 256
    
    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=6,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        # num_train_epochs=1,  # Set to 1 for a full training run
        max_steps=250,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",  # Can use Weights & Biases
        output_dir="outputs",
    )
    
    return training_args


def train_model(model, tokenizer, dataset, reward_funcs, training_args):
    """Train the model using GRPO."""
    print("Starting GRPO training...")
    print("You might have to wait 150 to 200 steps for any action.")
    print("You'll probably get 0 reward for the first 100 steps. Please be patient!")
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    return trainer


def test_model_before_training(model, tokenizer):
    """Test the model before GRPO training."""
    print("\n" + "="*50)
    print("Testing model BEFORE GRPO training:")
    print("="*50)
    
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": "Calculate pi."},
    ], tokenize=False, add_generation_prompt=True)
    
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    
    output = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0].outputs[0].text
    
    print("Model output before training:")
    print(output)
    print("\n")


def test_model_after_training(model, tokenizer, SYSTEM_PROMPT):
    """Test the model after GRPO training."""
    print("\n" + "="*50)
    print("Testing model AFTER GRPO training:")
    print("="*50)
    
    # Save the LoRA
    print("Saving LoRA...")
    model.save_lora("grpo_saved_lora")
    
    # Test with the trained LoRA
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Calculate pi."},
    ], tokenize=False, add_generation_prompt=True)
    
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora("grpo_saved_lora"),
    )[0].outputs[0].text
    
    print("Model output after training:")
    print(output)
    print("\n")


def save_model_options(model, tokenizer):
    """Provide options for saving the model in different formats."""
    print("\n" + "="*50)
    print("Model saving options (commented out by default):")
    print("="*50)
    
    # Uncomment the desired saving method:
    
    # Merge to 16bit
    # model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    # model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_16bit", token="")
    
    # Merge to 4bit
    # model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
    # model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_4bit", token="")
    
    # Just LoRA adapters
    # model.save_pretrained("model")
    # tokenizer.save_pretrained("model")
    # model.push_to_hub("hf/model", token="")
    # tokenizer.push_to_hub("hf/model", token="")
    
    # Save to GGUF formats
    # model.save_pretrained_gguf("model", tokenizer)  # 8bit Q8_0
    # model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")  # 16bit GGUF
    # model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")  # q4_k_m GGUF
    
    # Save to multiple GGUF options
    # model.push_to_hub_gguf(
    #     "hf/model",  # Change hf to your username!
    #     tokenizer,
    #     quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
    #     token="",
    # )
    
    print("Model saving options are commented out in the code.")
    print("Uncomment the desired method to save your model.")


def main():
    """Main function to run the entire GRPO training pipeline."""
    print("Starting Llama 3.1 8B GRPO Training Pipeline")
    print("="*60)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training will be very slow on CPU.")
        print("Consider using a GPU for better performance.")
    
    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # Setup data
        dataset, SYSTEM_PROMPT, extract_xml_answer = setup_data()
        
        # Setup reward functions
        reward_funcs = setup_reward_functions(extract_xml_answer)
        
        # Setup training configuration
        training_args = setup_training_config(1024)  # max_seq_length
        
        # Test model before training
        test_model_before_training(model, tokenizer)
        
        # Train the model
        trainer = train_model(model, tokenizer, dataset, reward_funcs, training_args)
        
        # Test model after training
        test_model_after_training(model, tokenizer, SYSTEM_PROMPT)
        
        # Show saving options
        save_model_options(model, tokenizer)
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("Check the 'outputs' directory for training logs and checkpoints.")
        print("Check 'grpo_saved_lora' directory for the saved LoRA adapter.")
        print("="*60)
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main() 