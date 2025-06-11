import sys
sys.path.append('.')
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    TrainingArguments, Trainer, set_seed)
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


'''
Configs
'''
class CFG:
    EXPERIMENT_NAME = 'train_token_scheduler'
    MODEL_PATH = 'answerdotai/ModernBERT-base'
    DEBUG = False
    SEED = 2025
    # SFT settings
    DATASET_NUM_PROC = 16
    BATCH_SIZE = 32
    GRAD_ACCUM = 1
    GRAD_CHECKPOINT = False
    NUM_TRAIN_EPOCHS = 3
    SAVE_STEPS = 100
    EVAL_STEPS = 100
    LOG_STEP = 10
    LR = 1e-5


set_seed(CFG.SEED)


training_args = TrainingArguments(
    run_name=CFG.EXPERIMENT_NAME,
    learning_rate=CFG.LR,
    bf16=True,
    per_device_train_batch_size=CFG.BATCH_SIZE,
    per_device_eval_batch_size=CFG.BATCH_SIZE*2,
    gradient_accumulation_steps=CFG.GRAD_ACCUM,
    gradient_checkpointing=CFG.GRAD_CHECKPOINT,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    dataloader_num_workers=CFG.DATASET_NUM_PROC,
    num_train_epochs=CFG.NUM_TRAIN_EPOCHS,
    # save_steps=CFG.SAVE_STEPS,
    eval_strategy="steps",
    eval_steps=CFG.EVAL_STEPS,
    do_eval=True,
    logging_steps=CFG.LOG_STEP,
    optim='adamw_torch',
    lr_scheduler_type="cosine",
    report_to="none" if CFG.DEBUG else "wandb",
    output_dir=f"ft_models/{CFG.EXPERIMENT_NAME}",
    logging_dir=f"output/{CFG.EXPERIMENT_NAME}",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    # greater_is_better=False,
    max_grad_norm=1.0,
    save_total_limit=1,
)

model = AutoModelForSequenceClassification.from_pretrained(
    CFG.MODEL_PATH, num_labels=1, problem_type='regression')
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_PATH, use_fast=True)


def preprocess_func(sample):
    global tokenizer
    return tokenizer(sample['text'], max_length=256, padding=True, truncation=True)


def preprocess_dataset(sample):
    return {
        'text': sample['problem'],
        'label': float(sample['mean_completion_length'] / 12800.)
    }


dataset = load_dataset('RabotniKuma/Fast-Math-R1-Token-Scheduler')['train']
dataset = dataset.map(preprocess_dataset)
dataset = dataset.remove_columns(['problem'])
dataset = dataset.map(preprocess_func, batched=True)
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=CFG.SEED)


'''
Training
'''
def preprocess_function(sample):
    return tokenizer(sample['problem'], max_length=256, padding=True, truncation=True)


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
#     auc = roc_auc_score(labels, probs)
#     return {"roc_auc": auc}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)
trainer.train()
