
#!/bin/bash

MODEL_TYPE=llava  #llama3-8b

PRETRAIN_DIR=pretrain-$MODEL_TYPE-mlp


mkdir -p ./checkpoints-$MODEL_TYPE

deepspeed bunny/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path /path/to/base_llm_model \
    --model_type $MODEL_TYPE \
    --version vicuna_v1 \
    --layer_selection "1,5,9,13,19,23,27,31"  # Pass the list as a string argument for activating layers
    --data_path ./data/finetune/llava_665k.json \ # Bunny_695k.json
    --image_folder ./data/finetune/images \
    --vision_tower /path/to/clip-336 \     #siglip-so400m-patch14-384
    --pretrain_mm_mlp_adapter ./checkpoints-pretrain/$PRETRAIN_DIR/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 ./checkpoints-$MODEL_TYPE/log.txt
