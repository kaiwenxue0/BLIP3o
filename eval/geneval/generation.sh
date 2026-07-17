#!/bin/bash

cd /home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/remote/BLIP3o/eval/geneval

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/notebook/code/group/xuekaiwen/.cache/huggingface


MODEL="/home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/remote/BLIP3o/output_train/checkpoint-11000"



# Total number of GPUs/chunks.
N_CHUNKS=2

# Launch processes in parallel for each GPU/chunk.
for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python generate.py --model "$MODEL" --index $i --n_chunks $N_CHUNKS > "generate_${i}_$(TZ=Asia/Taipei date +%-m.%-d_%H%M%S).log" 2>&1 &
done

# Wait for all background processes to finish.
wait
echo "All background processes finished."


