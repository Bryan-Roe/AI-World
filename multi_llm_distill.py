import os
import json
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset

# HuggingFace Transformers for student model fine-tuning
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ai_training', 'language_model', 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'ai_training', 'language_model', 'models', 'distill_student')
COLLAB_PATH = os.path.join(DATA_DIR, 'collab.jsonl')

CONFIG = {
    'base_model': os.environ.get('STUDENT_BASE_MODEL', 'distilgpt2'),
    'max_length': 512,
    'epochs': 3,
    'batch_size': 2,
    'lr': 5e-5,
    'warmup_steps': 100,
    'save_steps': 0,
    'logging_steps': 10,
    'prompt_prefix': 'User:',
    'answer_prefix': '\nAssistant:',
}


def load_collab(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Collab dataset not found: {path}\nRun collect_collab_data.py first.")
    data: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get('prompt', '')
            best = obj.get('best', '')
            if prompt and best:
                data.append({'prompt': prompt, 'best': best})
    return data


@dataclass
class CollabExample:
    prompt: str
    best: str


class CollabDataset(Dataset):
    def __init__(self, examples: List[CollabExample], tokenizer: AutoTokenizer, max_length: int, prompt_prefix: str, answer_prefix: str):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_prefix = prompt_prefix
        self.answer_prefix = answer_prefix

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_text = f"{self.prompt_prefix} {ex.prompt}{self.answer_prefix} "
        full_text = prompt_text + ex.best

        # Tokenize full sequence
        enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = enc['input_ids'][0]
        attention_mask = enc['attention_mask'][0]

        # Mask loss on prompt part: labels = -100 for prompt tokens
        prompt_enc = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length, return_tensors='pt')
        prompt_len = prompt_enc['input_ids'].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading collab dataset from {COLLAB_PATH}")
    raw = load_collab(COLLAB_PATH)
    examples = [CollabExample(prompt=r['prompt'], best=r['best']) for r in raw]
    print(f"Loaded {len(examples)} examples")

    print(f"Loading tokenizer/model: {CONFIG['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(CONFIG['base_model'])

    dataset = CollabDataset(examples, tokenizer, CONFIG['max_length'], CONFIG['prompt_prefix'], CONFIG['answer_prefix'])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=CONFIG['batch_size'],
        num_train_epochs=CONFIG['epochs'],
        learning_rate=CONFIG['lr'],
        warmup_steps=CONFIG['warmup_steps'],
        logging_steps=CONFIG['logging_steps'],
        save_steps=CONFIG['save_steps'],
        bf16=torch.cuda.is_available(),
        fp16=False,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete. Saving model...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Saved distilled student to {MODEL_DIR}")


if __name__ == '__main__':
    main()
