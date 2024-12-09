# use ylecun/mnist for automatic mnist download
# use ernestchu/emnist-digits for automatic emnist-digits download
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" python train_unconditional-v2.py \
    --dataset_name "ernestchu/emnist-digits" \
    --output_dir "./output/uncnd_no_aug" \
    --resolution 32 \
    --train_batch_size 128 \
    --num_epochs 30 \
    --learning_rate 1e-4 \
    --ddpm_num_steps 1000 \
    --ddpm_beta_schedule "linear" \
    --mixed_precision no \
    --gradient_accumulation_steps 1 \
    --printed_digits_dir "./printed_digits" \
    --replacement_prob 0.3 \
    --save_images_epochs 1 \
    --save_model_epochs 1 \
    --checkpointing_steps 100000 \
    --disable_class_conditioning \
    --overwrite_output_dir
    # --augment_printed \
