export TRAIN_FILE=data/sarcasm.txt
export TEST_FILE=data/sarcasm.txt

python soa_tuning.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --per_gpu_train_batch_size 1 \
    --num_train_epochs 3 
