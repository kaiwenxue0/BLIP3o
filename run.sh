# 在代码中替换路径
# vim remote/BLIP3o/blip3o/model/multimodal_encoder/eva_clip/eva_vit.py
# cache_dir = "/mnt/workspace/xuekaiwen/data/jiuhai/eva_clip_vision_tower"

# 建议的本地可写路径
mkdir -p /home/notebook/code/group/xuekaiwen/{runs,logs,tmp,.cache/huggingface/{hub,datasets,transformers},.cache/torch,wandb}
#!/bin/bash


conda activate py310



#  (qwen 3 8B + siglip2 428M + mlp2x_gelu projector 6.56M) + (evaclip 4.35B + sdxl 2.6B + vae 80M + dit 1.36B)
export OUTPUT_FOLDER=/home/notebook/code/group/xuekaiwen/mask_diffusion/nanoMDM/remote/BLIP3o/output_train
export IMG_FOLDER=/mnt/workspace/xuekaiwen/data/BLIP3o/BLIP3o-Pretrain-Long-Caption
export journeyDB_folder=/mnt/workspace/xuekaiwen/data/BLIP3o/BLIP3o-Pretrain-JourneyDB

export model_name_or_path=/mnt/workspace/xuekaiwen/data/Qwen/Qwen3-1.7B
export vision_tower=/mnt/workspace/xuekaiwen/data/google/siglip2-so400m-patch16-512
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

torchrun --nproc_per_node=8 \
    blip3o/train/train_mem.py \
    --deepspeed ./deepspeed_scripts/zero1.json \
    --model_name_or_path ${model_name_or_path} \
    --version qwen \
    --data_type "mix" \
    --image_folder ${IMG_FOLDER} \
    --journeyDB_folder  ${journeyDB_folder} \
    --vision_tower ${vision_tower} \
    --freeze_backbone False \
    --gen_vision_tower ${gen_vision_tower} \
    --gen_projector_type mlp2x_gelu \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_FOLDER} \
    --num_train_epochs 10 \
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
    --model_max_length 2048 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --gen_pooling early_pool2d_4 \
    --n_query 64 \
    --n_und_query 1024 \
    --report_to none \
    --run_name blip3o_qwen3_siglip2 \
    > train.log 2>&1


#add
export IMAGE_CAHCE_DIR=/home/notebook/code/group/xuekaiwen/.cache/huggingface/datasets/webdataset/default-30dae044d7267163/0.0.0/8baf059d8b05687d95d8bae817c9fc387eb06339b6bf00740e46581c077c783e

 --image_cache_dir ${IMAGE_CAHCE_DIR}

#  (qwen 2.5 VL 7B Instruct) + (evaclip 4.35B + sdxl 2.6B + vae 80M + dit 1.36B)
#