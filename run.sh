#!/bin/bash

conda activate py310

export OUTPUT_FOLDER=/home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/remote/BLIP3o/output_train
export IMG_FOLDER=/home/notebook/code/group/xuekaiwen/data/BLIP3o/BLIP3o-Pretrain-Long-Caption
export journeyDB_folder=/home/notebook/code/group/xuekaiwen/data/BLIP3o/BLIP3o-Pretrain-JourneyDB

export model_name_or_path=/home/notebook/code/group/xuekaiwen/data/Qwen/Qwen2.5-VL-7B-Instruct
export gen_vision_tower=eva-clip-E-14-plus

export TORCH_HOME=/home/notebook/code/group/xuekaiwen/.cache/torch
export WANDB_DIR=/home/notebook/code/group/xuekaiwen/wandb
export WANDB_CACHE_DIR=/home/notebook/code/group/xuekaiwen/wandb
export TMPDIR=/home/notebook/code/group/xuekaiwen/tmp
export HF_HOME=/home/notebook/code/group/xuekaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_ENDPOINT=https://hf-mirror.com
export RUN_NAME=blip3o_qwen_vl_7b

torchrun --nproc_per_node=8 \
    blip3o/train/train_mem.py \
    --deepspeed ./deepspeed_scripts/zero1.json \
    --model_name_or_path ${model_name_or_path}  \
    --version qwen \
    --data_type "mix" \
    --image_folder ${IMG_FOLDER} \
    --gen_vision_tower ${gen_vision_tower} \
    --gen_projector_type mlp2x_gelu \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_FOLDER} \
    --num_train_epochs 30 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 0 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --model_max_length 512 \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --gen_pooling early_pool2d_4 \
    --n_query 64 \
    --n_und_query 0 \
    --report_to none \
    --run_name ${RUN_NAME}



echo "[Starfire] NODE_RANK=${NODE_RANK} NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[Starfire] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} RDZV_ID=${RDZV_ID}"
nvidia-smi || true

cd /home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/remote/BLIP3o
pip install -e .
torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  blip3o/train/train_mem.py \
 --deepspeed ./deepspeed_scripts/zero1.json \
    --model_name_or_path ${model_name_or_path} \
    --version qwen \
    --data_type "mix" \
    --image_folder ${IMG_FOLDER} \
    --gen_vision_tower ${gen_vision_tower} \
    --gen_projector_type mlp2x_gelu \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_FOLDER} \
    --num_train_epochs 30 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --model_max_length 512 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --gen_pooling early_pool2d_4 \
    --n_query 64 \
    --n_und_query 0 \
    --run_name ${RUN_NAME} \
    --image_cache_dir ${IMAGE_CAHCE_DIR} \
    > "train_${RUN_NAME}_$(TZ=Asia/Taipei date +%-m.%-d_%H%M%S).log" 2>&1



# 8.20 
echo "[Starfire] NODE_RANK=${NODE_RANK} NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[Starfire] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} RDZV_ID=${RDZV_ID}"
nvidia-smi || true

cd /home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/remote/BLIP3o
pip install -e .

if [ -z "${NPROC_PER_NODE+x}" ]; then
    export NPROC_PER_NODE=2
    export MASTER_ADDR=localhost
    export MASTER_PORT=12345
fi

export OUTPUT_FOLDER=/home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/remote/BLIP3o/output_train
export IMG_FOLDER=/home/notebook/code/group/xuekaiwen/data/BLIP3o/BLIP3o-Pretrain-Long-Caption
export journeyDB_folder=/home/notebook/code/group/xuekaiwen/data/BLIP3o/BLIP3o-Pretrain-JourneyDB

export model_name_or_path=/home/notebook/code/group/xuekaiwen/data/Qwen/Qwen2.5-VL-7B-Instruct
export gen_vision_tower=eva-clip-E-14-plus

export TORCH_HOME=/home/notebook/code/group/xuekaiwen/.cache/torch
export WANDB_DIR=/home/notebook/code/group/xuekaiwen/wandb
export WANDB_CACHE_DIR=/home/notebook/code/group/xuekaiwen/wandb
export TMPDIR=/home/notebook/code/group/xuekaiwen/tmp
export HF_HOME=/home/notebook/code/group/xuekaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_ENDPOINT=https://hf-mirror.com
export RUN_NAME=blip3o_qwen_vl_7b


torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  blip3o/train/train_mem.py \
  --deepspeed ./deepspeed_scripts/zero1.json \
  --model_name_or_path ${model_name_or_path} \
  --version qwen \
  --data_type "mix" \
  --image_folder ${IMG_FOLDER} \
  --gen_vision_tower ${gen_vision_tower} \
  --gen_projector_type mlp2x_gelu \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir ${OUTPUT_FOLDER} \
  --num_train_epochs 30 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.003 \
  --lr_scheduler_type "cosine_with_min_lr" \
  --lr_scheduler_kwargs '{"min_lr":1e-5}' \
  --model_max_length 512 \
  --tf32 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --gen_pooling early_pool2d_4 \
  --n_query 64 \
  --n_und_query 0 \
  --run_name ${RUN_NAME} \
  > "train_${RUN_NAME}_$(TZ=Asia/Taipei date +%-m.%-d_%H%M%S).log" 2>&1

# remove the rng state files
sudo find . -maxdepth 2 -type f -name "rng_state_*.pth" -print -delete

# export HF_HOME=/home/notebook/code/group/xuekaiwen/.cache/huggingface
# export HF_HUB_CACHE=$HF_HOME/hub
# export HF_DATASETS_CACHE=$HF_HOME/datasets
# export HF_ENDPOINT=https://hf-mirror.com
# python data_test.py \
#   --image_folder /home/notebook/code/group/xuekaiwen/data/BLIP3o/BLIP3o-Pretrain-Long-Caption \
#   --sample 8

export HF_HOME=/home/notebook/code/group/xuekaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ENDPOINT=https://hf-mirror.com
python data_test.py \
  --image_folder /home/notebook/code/group/xuekaiwen/data/BLIP3o/BLIP3o-Pretrain-Short-Caption \
  --sample 64 > data_test.log

export HF_HOME=/home/notebook/code/group/xuekaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ENDPOINT=https://hf-mirror.com
python data_test.py \
  --video_icedit_folder /home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/data/OmniGen2/X2I2/video_icedit/webds_shards \
  --sample 64 > data_test.log

# data test
export HF_HOME=/home/notebook/code/group/xuekaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ENDPOINT=https://hf-mirror.com
export SAVE_DIR=data_test_video_icedit
python data_test.py \
 --video_icedit_folder /home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/data/OmniGen2/X2I2/video_icedit/webds_shards \
 --sample 64 > ${SAVE_DIR}.log

export HF_HOME=/home/notebook/code/group/xuekaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ENDPOINT=https://hf-mirror.com
export SAVE_DIR=data_test_video_icgen_0
python data_test.py \
 --video_icgen_folder /home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/data/OmniGen2/X2I2/video_icgen/icgen_mv_0/webds_shards \
 --sample 64 > ${SAVE_DIR}.log

python data_test.py \
  --image_folder /home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/data/BLIP3o/BLIP3o-Pretrain-Long-Caption \
  --sample 64 > data_test.log

python data_test.py \
  --image_folder /home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/data/BLIP3o/BLIP3o-Pretrain-Short-Caption \
  --sample 64 > data_test.log

python data_test.py \
  --image_folder /home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/data/BLIP3o/BLIP3o-60k \
  --sample 58859 > data_test_BLIP3o-60k.log


#train on blip3o
echo "[Starfire] NODE_RANK=${NODE_RANK} NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[Starfire] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} RDZV_ID=${RDZV_ID}"
nvidia-smi || true

cd /home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/remote/BLIP3o
pip install -e .

if [ -z "${NPROC_PER_NODE+x}" ]; then
    export NPROC_PER_NODE=2
    export MASTER_ADDR=localhost
    export MASTER_PORT=12345
fi

export OUTPUT_FOLDER=/home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/remote/BLIP3o/output_train.debug
export video_icedit_folder=/home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/data/OmniGen2/X2I2/video_icedit/webds_shards

export model_name_or_path=/home/notebook/code/group/xuekaiwen/data/Qwen/Qwen2.5-VL-7B-Instruct
export gen_vision_tower=eva-clip-E-14-plus

export TORCH_HOME=/home/notebook/code/group/xuekaiwen/.cache/torch
export WANDB_DIR=/home/notebook/code/group/xuekaiwen/wandb
export WANDB_CACHE_DIR=/home/notebook/code/group/xuekaiwen/wandb
export TMPDIR=/home/notebook/code/group/xuekaiwen/tmp
export HF_HOME=/home/notebook/code/group/xuekaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_ENDPOINT=https://hf-mirror.com
export RUN_NAME=blip3o_qwen_vl_7b

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  blip3o/train/train_mem.py \
  --deepspeed ./deepspeed_scripts/zero1.json \
  --model_name_or_path ${model_name_or_path} \
  --version qwen \
  --data_type "mix" \
  --video_icedit_folder ${video_icedit_folder} \
  --gen_vision_tower ${gen_vision_tower} \
  --gen_projector_type mlp2x_gelu \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir ${OUTPUT_FOLDER} \
  --num_train_epochs 30 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.003 \
  --lr_scheduler_type "cosine_with_min_lr" \
  --lr_scheduler_kwargs '{"min_lr":1e-5}' \
  --model_max_length 512 \
  --tf32 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --gen_pooling early_pool2d_4 \
  --n_query 64 \
  --n_und_query 0 \
  --run_name ${RUN_NAME} \
  > "train_${RUN_NAME}_$(TZ=Asia/Taipei date +%-m.%-d_%H%M%S).log" 2>&1





if [ -z "${NPROC_PER_NODE+x}" ]; then
    export NPROC_PER_NODE=2
    export MASTER_ADDR=localhost
    export MASTER_PORT=12345
fi

export OUTPUT_FOLDER=/home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/remote/BLIP3o/output_train.debug
export video_icedit_folder=/home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/data/OmniGen2/X2I2/video_icedit/webds_shards

export model_name_or_path=/home/notebook/code/group/xuekaiwen/data/Qwen/Qwen2.5-VL-7B-Instruct
export gen_vision_tower=eva-clip-E-14-plus

export TORCH_HOME=/home/notebook/code/group/xuekaiwen/.cache/torch
export WANDB_DIR=/home/notebook/code/group/xuekaiwen/wandb
export WANDB_CACHE_DIR=/home/notebook/code/group/xuekaiwen/wandb
export TMPDIR=/home/notebook/code/group/xuekaiwen/tmp
export HF_HOME=/home/notebook/code/group/xuekaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_ENDPOINT=https://hf-mirror.com
export RUN_NAME=blip3o_qwen_vl_7b

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  blip3o/train/train_mem.py \
  --deepspeed ./deepspeed_scripts/zero1.json \
  --model_name_or_path ${model_name_or_path} \
  --version qwen \
  --data_type "mix" \
  --video_icedit_folder ${video_icedit_folder} \
  --gen_vision_tower ${gen_vision_tower} \
  --gen_projector_type mlp2x_gelu \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir ${OUTPUT_FOLDER} \
  --num_train_epochs 30 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.003 \
  --lr_scheduler_type "cosine_with_min_lr" \
  --lr_scheduler_kwargs '{"min_lr":1e-5}' \
  --model_max_length 512 \
  --tf32 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --gen_pooling early_pool2d_4 \
  --n_query 64 \
  --n_und_query 256 \
  --run_name ${RUN_NAME} \
  > "train_${RUN_NAME}_$(TZ=Asia/Taipei date +%-m.%-d_%H%M%S).log" 2>&1