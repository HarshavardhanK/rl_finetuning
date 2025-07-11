#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Llama 3.2 3B GRPO Training Script

This script implements GRPO (Group Relative Policy Optimization) training for
Llama 3.2 3B Instruct model on GSM8K dataset for mathematical reasoning.

Original notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb
"""

import os
import re
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from safetensors import safe_open


def setup_model_and_tokenizer():
    """Setup the model and tokenizer with LoRA configuration."""
    print("Setting up model and tokenizer...")
    
    max_seq_length = 2048  # Can increase for longer reasoning traces
    lora_rank = 64  # Larger rank = smarter, but slower
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # False for LoRA 16bit
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
    
    return model, tokenizer, max_seq_length


def setup_data():
    """Setup the GSM8K dataset and data processing functions."""
    print("Setting up dataset...")
    
    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    # Define reasoning format symbols
    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"
    
    system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
    
    def extract_hash_answer(text):
        """Extract answer from hash format."""
        if "####" not in text:
            return None
        return text.split("####")[1].strip()
    
    # Map the dataset
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })
    
    return dataset, system_prompt, reasoning_start, reasoning_end, solution_start, solution_end


def setup_reward_functions(reasoning_start, reasoning_end, solution_start, solution_end):
    """Setup reward functions for GRPO training."""
    print("Setting up reward functions...")
    
    # Create regex format to match the reasoning sections and answers
    match_format = re.compile(
        rf"^[\s]{{0,}}"\
        rf"{reasoning_start}.+?{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end}"\
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Create regex for number extraction
    match_numbers = re.compile(
        solution_start + r".*?([\d\.\,]{1,})",
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Global variables for printing control
    global PRINTED_TIMES
    PRINTED_TIMES = 0
    global PRINT_EVERY_STEPS
    PRINT_EVERY_STEPS = 5
    
    def match_format_exactly(completions, **kwargs):
        """Reward function for exact format matching."""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Match if format is seen exactly!
            if match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        return scores
    
    def match_format_approximately(completions, **kwargs):
        """Reward function for approximate format matching."""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Count how many keywords are seen - we penalize if too many!
            # If we see 1, then plus some points!
            score += 0.5 if response.count(reasoning_start) == 1 else -1.0
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(solution_start) == 1 else -1.0
            score += 0.5 if response.count(solution_end) == 1 else -1.0
            scores.append(score)
        return scores
    
    def check_answer(prompts, completions, answer, **kwargs):
        """Reward function for answer correctness."""
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        
        extracted_responses = [
            guess.group(1)
            if (guess := match_format.search(r)) is not None else None
            for r in responses
        ]
        
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(0)
                continue
            # Correct answer gets 3 points!
            if guess == true_answer:
                score += 3.0
            # Match if spaces are seen, but less reward
            elif guess.strip() == true_answer.strip():
                score += 1.5
            else:
                # We also reward it if the answer is close via ratios!
                # Ie if the answer is within some range, reward it!
                try:
                    ratio = float(guess) / float(true_answer)
                    if ratio >= 0.9 and ratio <= 1.1:
                        score += 1.0
                    elif ratio >= 0.8 and ratio <= 1.2:
                        score += 0.5
                    else:
                        score -= 1.5  # Penalize wrong answers
                except:
                    score -= 1.5  # Penalize
            scores.append(score)
        return scores
    
    def check_numbers(prompts, completions, answer, **kwargs):
        """Reward function for number extraction and correctness."""
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        
        extracted_responses = [
            guess.group(1)
            if (guess := match_numbers.search(r)) is not None else None
            for r in responses
        ]
        
        scores = []
        # Print only every few steps
        global PRINTED_TIMES
        global PRINT_EVERY_STEPS
        if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
            print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", 
                  f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        PRINTED_TIMES += 1
        
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0)
                continue
            # Convert to numbers
            try:
                true_answer = float(true_answer.strip())
                # Remove commas like in 123,456
                guess = float(guess.strip().replace(",", ""))
                scores.append(1.5 if guess == true_answer else -0.5)
            except:
                scores.append(0)
                continue
        return scores
    
    return [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ]


def setup_training_config(max_seq_length):
    """Setup GRPO training configuration."""
    print("Setting up training configuration...")
    
    # Get the maximum prompt length
    max_prompt_length = 287 + 1  # + 1 just in case!
    
    training_args = GRPOConfig(
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Increase to 4 for smoother training
        num_generations=4,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        # num_train_epochs=1,  # Set to 1 for a full training run
        max_steps=500,
        save_steps=250,
        max_grad_norm=1.0,
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
        {"role": "user", "content": "What is the sqrt of 101?"},
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


def test_model_after_training(model, tokenizer, system_prompt):
    """Test the model after GRPO training."""
    print("\n" + "="*50)
    print("Testing model AFTER GRPO training:")
    print("="*50)
    
    # Save the LoRA
    print("Saving LoRA...")
    model.save_lora("grpo_saved_lora")
    
    # Verify LoRA is actually trained
    print("Verifying LoRA training...")
    try:
        tensors = {}
        with safe_open("grpo_saved_lora/adapter_model.safetensors", framework="pt") as f:
            # Verify both A and B are non zero
            for key in f.keys():
                tensor = f.get_tensor(key)
                n_zeros = (tensor == 0).sum() / tensor.numel()
                assert(n_zeros.item() != tensor.numel())
        print("LoRA verification successful!")
    except Exception as e:
        print(f"LoRA verification failed: {e}")
    
    # Test with the trained LoRA
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is the sqrt of 101?"},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # Must add for generation
        tokenize=False,
    )
    
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
    print("Starting Advanced Llama 3.2 3B GRPO Training Pipeline")
    print("="*60)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training will be very slow on CPU.")
        print("Consider using a GPU for better performance.")
    
    try:
        # Setup model and tokenizer
        model, tokenizer, max_seq_length = setup_model_and_tokenizer()
        
        # Setup data
        dataset, system_prompt, reasoning_start, reasoning_end, solution_start, solution_end = setup_data()
        
        # Setup reward functions
        reward_funcs = setup_reward_functions(reasoning_start, reasoning_end, solution_start, solution_end)
        
        # Setup training configuration
        training_args = setup_training_config(max_seq_length)
        
        # Test model before training
        test_model_before_training(model, tokenizer)
        
        # Train the model
        trainer = train_model(model, tokenizer, dataset, reward_funcs, training_args)
        
        # Test model after training
        test_model_after_training(model, tokenizer, system_prompt)
        
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