export CUDA_VISIBLE_DEVICES=0

# RLFT
torchrun --nproc_per_node=1 --master_port=20001  run_unlearn_api_lora.py   \
    --target_model_name_or_path ../models/deepseek-coder-1.3B \
    --per_device_train_batch_size 1     \
    --do_unlearn  \
    --output_dir ./output \
    --overwrite_output_dir     \
    --num_train_epochs 5    \
    --logging_steps 1     \
    --learning_rate 2e-5     \
    --warmup_ratio 0.03 \
    --overwrite_cache \
    --save_strategy epoch \
    --save_total_limit 5 \
    --bf16 True \
    --tf32 True \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --domain ds-forget \
    --gradient_accumulation_steps 85 \
    --unlearn_method random_label --completely_random True

#seven machine unlearning methods
methods=(
    "GA" 
    "GD" 
    "KL" 
    "DPO" 
    "NPO"
    "SimNPO"
    "PROD"
)

for method in "${methods[@]}"
do

    port=$((20000 + RANDOM % 1000))

    output_dir="./output/${method}"

    echo "======================================"
    echo "Running unlearn method: ${method}"
    echo "Output: ${output_dir}"
    echo "Port: ${port}"
    echo "======================================"


    torchrun \
        --nproc_per_node=1 \
        --master_port=${port} \
        run_unlearn_api_lora.py \
        --target_model_name_or_path ../models/deepseek-coder-1.3B \
        --per_device_train_batch_size 1 \
        --do_unlearn \
        --output_dir ./output \
        --overwrite_output_dir \
        --num_train_epochs 1 \
        --logging_steps 1 \
        --learning_rate 2e-5 \
        --warmup_ratio 0.03 \
        --overwrite_cache \
        --save_strategy epoch \
        --save_total_limit 5 \
        --bf16 True \
        --tf32 True \
        --weight_decay 0. \
        --lr_scheduler_type cosine \
        --domain ds-forget \
        --gradient_accumulation_steps 85 \
        --unlearn_method ${method}


    echo "${method} finished!"
    echo

done


echo "All unlearning experiments finished!"