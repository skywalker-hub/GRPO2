stage 1:
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --num_processes 8 \
experiments/train_first_stage.py


stage2:
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --num_processes 8 \
experiments/train_second_stage.p
