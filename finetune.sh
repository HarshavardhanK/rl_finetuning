export CUDA_VISIBLE_DEVICES=1,5
tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed --config llama2-7b-lora_config.yaml