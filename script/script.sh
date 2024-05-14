python train_text_to_image_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --dataset_name="..\..\data\hamster" \
    --dataloader_num_workers=0 \
    --resolution=512 \
    --center_crop \
    --random_flip \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=240 \
    --learning_rate=1e-04 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --output_dir="..\..\output" \
    --report_to=wandb \
    --use_8bit_adam --adam_beta1=0.9 --adam_weight_decay=1e-2 \
    --validation_prompt="hamster" \
    --seed=1337 \