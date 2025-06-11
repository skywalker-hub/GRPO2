import sys
sys.path.append('.')
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer, get_kbit_device_map
import pandas as pd
import numpy as np
from pathlib import Path


'''
Configs
'''
class CFG:
    EXPERIMENT_NAME = 'train_first_stage'
    MODEL_PATH = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    DEBUG = False
    SEED = 2025
    # SFT settings
    DATASET_NUM_PROC = 16
    BATCH_SIZE = 1
    GRAD_ACCUM = 8
    PACKING = True
    EVAL_PACKING = True
    MAX_SEQ_LENGTH = 24000
    GRAD_CHECKPOINT = True
    NUM_TRAIN_EPOCHS = 20
    SAVE_STEPS = 50
    EVAL_STEPS = None
    LOG_STEP = 1
    LR = 1e-5
    # PEFT settings
    USE_PEFT = False
    USE_QLORA = False
    LORA_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    LORA_R = 32
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05


set_seed(CFG.SEED)


training_args = SFTConfig(
    run_name=CFG.EXPERIMENT_NAME,
    learning_rate=CFG.LR,
    bf16=True,
    per_device_train_batch_size=CFG.BATCH_SIZE,
    gradient_accumulation_steps=CFG.GRAD_ACCUM,
    gradient_checkpointing=CFG.GRAD_CHECKPOINT,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    dataset_num_proc=CFG.DATASET_NUM_PROC,
    packing=CFG.PACKING,
    eval_packing=CFG.EVAL_PACKING,
    max_seq_length=CFG.MAX_SEQ_LENGTH,
    num_train_epochs=CFG.NUM_TRAIN_EPOCHS,
    save_steps=CFG.SAVE_STEPS,
    eval_strategy="steps" if CFG.EVAL_STEPS is not None else "no",
    eval_steps=CFG.EVAL_STEPS,
    per_device_eval_batch_size=CFG.BATCH_SIZE,
    do_eval=False,
    logging_steps=CFG.LOG_STEP,
    optim='paged_adamw_8bit',
    lr_scheduler_type="cosine",
    report_to="none" if CFG.DEBUG else "wandb",
    output_dir=f"ft_models/{CFG.EXPERIMENT_NAME}",
    logging_dir=f"output/{CFG.EXPERIMENT_NAME}",
    save_strategy="steps",
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
    save_total_limit=1,
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
    system_prompt = 'Please reason step by step, and put your final answer within \\boxed{{}}.'
    return {
        'messages': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample['problem']},
            {"role": "assistant", "content": sample['generation']},
        ]
    }


dataset = load_dataset('RabotniKuma/Fast-Math-R1-SFT')['train']
dataset = dataset.map(create_prompt)


'''
Training
'''
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_PATH)
tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
training_args.model_init_kwargs = dict(
    quantization_config=quant_config if CFG.USE_QLORA else None,
    trust_remote_code=True,
    use_cache=False if CFG.GRAD_CHECKPOINT else True,
    device_map=get_kbit_device_map() if CFG.USE_QLORA else None,
    attn_implementation="flash_attention_2",
    torch_dtype="bfloat16",
)

if __name__ == '__main__':
    trainer = SFTTrainer(
        model=CFG.MODEL_PATH,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config if CFG.USE_PEFT else None,
        processing_class=tokenizer,
    )
    if len(list(Path(f'ft_models/{CFG.EXPERIMENT_NAME}').glob('checkpoint-*'))) > 0:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
