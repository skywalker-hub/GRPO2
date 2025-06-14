from datasets import Dataset, load_dataset
from transformers import BitsAndBytesConfig
from peft import LoraConfig
# from trl import GRPOConfig, GRPOTrainer
from finetune.openr1_mod.faster_grpo_trainer import FastGRPOTrainer as GRPOTrainer
from finetune.openr1_mod.configs import GRPOConfig, vLLMConfig
import pandas as pd
import numpy as np
from transformers import set_seed

import sys
sys.path.append('.')
from finetune.grpo_rewards import REWARD_FUNCS_REGISTRY, get_cosine_scaled_reward


'''
Configs
'''
class CFG:
    EXPERIMENT_NAME = 'train_second_stage'
    MODEL_PATH = 'sft_models/train_first_stage/checkpoint-350'
    DEBUG = False
    SEED = 2025
    # GRPO settings
    reward_funcs = [
        'format2',   # boxed before </think>
        'cosine',
        'length']
    USE_VLLM = True
    MAX_PROMPT_LENGTH = 512
    MAX_COMPLETION_LENGTH = 16384
    NUM_GENERATIONS = 8
    BATCH_SIZE = 2
    GRAD_ACCUM = 8
    GRAD_CHECKPOINT = True
    NUM_TRAIN_EPOCHS = 1
    MAX_STEPS = 40
    SAVE_STEPS = 5
    EVAL_STEPS = 5
    LOG_STEP = 1
    BETA = 0.04
    LR = 4e-6
    # PEFT settings
    USE_PEFT = False
    USE_QLORA = False
    LORA_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    LORA_R = 32
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05


set_seed(CFG.SEED)

training_args = GRPOConfig(
    run_name=CFG.EXPERIMENT_NAME,
    use_vllm=CFG.USE_VLLM,
    vllm_config=vLLMConfig(
        max_num_batched_tokens=CFG.MAX_COMPLETION_LENGTH,
        max_model_len=CFG.MAX_COMPLETION_LENGTH,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=4,
        enforce_eager=True,
        disable_log_stats=True,
        sampling_params_dict=dict(
            temperature=1.0,
            top_p=0.9,
            min_p=0.05,
            max_tokens=CFG.MAX_COMPLETION_LENGTH,
        )
    ),
    learning_rate=CFG.LR,
    beta=CFG.BETA,
    bf16=True,
    per_device_train_batch_size=CFG.BATCH_SIZE,
    gradient_accumulation_steps=CFG.GRAD_ACCUM,
    gradient_checkpointing=CFG.GRAD_CHECKPOINT,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    num_generations=CFG.NUM_GENERATIONS,
    max_prompt_length=CFG.MAX_PROMPT_LENGTH,
    max_completion_length=CFG.MAX_COMPLETION_LENGTH,
    num_train_epochs=CFG.NUM_TRAIN_EPOCHS,
    max_steps=CFG.MAX_STEPS,
    save_steps=CFG.SAVE_STEPS,
    eval_strategy="steps",
    eval_steps=CFG.EVAL_STEPS,
    per_device_eval_batch_size=CFG.BATCH_SIZE*2,
    do_eval=False,
    logging_steps=CFG.LOG_STEP,
    lr_scheduler_type="cosine",
    report_to="none" if CFG.DEBUG else "wandb",
    output_dir=f"ft_models/{CFG.EXPERIMENT_NAME}",
    logging_dir=f"output/{CFG.EXPERIMENT_NAME}",
    save_total_limit=2,
    save_strategy="steps",
    optim='paged_adamw_8bit'
)

peft_config = LoraConfig(
    r=CFG.LORA_R,
    lora_alpha=CFG.LORA_ALPHA,
    lora_dropout=CFG.LORA_DROPOUT,
    target_modules=CFG.LORA_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

quant_config = BitsAndBytesConfig(
    # load_in_8bit=True
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='bfloat16',
    bnb_4bit_use_double_quant=False,
)


def create_prompt(sample):
    system_prompt = (
        'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
        'You should think step-by-step. Return final answer within \\boxed{{}}.'
    )
    return {
        'prompt': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample['problem']}
        ]
    }


if __name__ == "__main__":
    dataset = load_dataset('sky/Fast-R1-GRPO')['train']
    dataset = dataset.map(create_prompt)

    training_args.model_init_kwargs = dict(
        quantization_config=quant_config if CFG.USE_QLORA else None,
        trust_remote_code=True,
        use_cache=False if CFG.GRAD_CHECKPOINT else True,
    )

    reward_funcs = []
    for func_name in CFG.reward_funcs:
        if func_name == 'cosine':
            reward_funcs.append(
                get_cosine_scaled_reward(
                    max_value_correct=1.0,
                    min_value_correct=0.1,
                    max_value_wrong=-0.1,
                    min_value_wrong=-1.0,
                    max_len=30000,
                    clip_len=True,
                ))
        else:
            reward_funcs.append(REWARD_FUNCS_REGISTRY[func_name])

    trainer = GRPOTrainer(
        model=CFG.MODEL_PATH,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config if CFG.USE_PEFT else None,
    )
    trainer.train()
