#!/usr/bin/env python3
"""
Minimal Language Model Training
No complex dependencies, just the essentials
"""

import os
import sys

print("[*] Preparing training environment...")

# Step 1: Check and install minimal dependencies
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
except ImportError:
    print("[!] Installing PyTorch...")
    os.system(f"{sys.executable} -m pip install torch --quiet")
    import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    print(f"[OK] Transformers loaded")
except ImportError:
    print("[!] Installing transformers...")
    os.system(f"{sys.executable} -m pip install transformers --quiet")
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

try:
    from datasets import Dataset
    print(f"[OK] Datasets loaded")
except ImportError:
    print("[!] Installing datasets...")
    os.system(f"{sys.executable} -m pip install datasets --quiet")
    from datasets import Dataset

try:
    from tqdm import tqdm
    print(f"[OK] tqdm loaded")
except ImportError:
    print("[!] Installing tqdm...")
    os.system(f"{sys.executable} -m pip install tqdm --quiet")
    from tqdm import tqdm

print("\n" + "="*70)
print("[*] LANGUAGE MODEL TRAINING")
print("="*70)

# Step 2: Prepare directories
os.makedirs("ai_training/language_model/data", exist_ok=True)
os.makedirs("ai_training/language_model/models", exist_ok=True)

# Step 3: Check training data
data_file = "ai_training/language_model/data/train.txt"
if not os.path.exists(data_file):
    print(f"\n[!] Creating sample training data...")
    sample_text = """
    The future of artificial intelligence is filled with possibilities.
    Machine learning is transforming how we solve problems.
    Deep learning networks can learn complex patterns from data.
    Neural networks are inspired by biological brains.
    Training data is crucial for building good models.
    The transformer architecture revolutionized natural language processing.
    Attention mechanisms allow models to focus on important parts of input.
    Pre-training on large datasets improves model performance.
    Fine-tuning adapts pre-trained models to specific tasks.
    Large language models can generate coherent text.
    """
    with open(data_file, 'w') as f:
        f.write(sample_text)
    print(f"[OK] Created {data_file}")

# Step 4: Load and tokenize data
print(f"\n[*] Loading training data...")
with open(data_file, 'r') as f:
    text = f.read()

if len(text) < 100:
    print("[!] Warning: Training data is very small (<100 chars)")

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize
print(f"[*] Tokenizing text ({len(text)} characters)...")
tokens = tokenizer(text, truncation=True, max_length=512, padding=True, return_tensors="pt")
print(f"[OK] Tokenized to {tokens['input_ids'].shape[0]} sequences")

# Create simple dataset
def create_dataset():
    # Split into windows
    input_ids = tokens['input_ids'].squeeze()
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    dataset_list = []
    for i in range(len(input_ids)):
        dataset_list.append({
            'input_ids': input_ids[i].tolist(),
            'labels': input_ids[i].tolist()
        })
    
    return Dataset.from_list(dataset_list)

dataset = create_dataset()
print(f"[OK] Dataset has {len(dataset)} examples")

if len(dataset) == 0:
    print("[!] Error: No training examples created")
    sys.exit(1)

# Step 5: Setup training
print(f"\n[*] Setting up training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[OK] Using device: {device}")

# Load model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.to(device)

print(f"[OK] Model loaded")
total_params = sum(p.numel() for p in model.parameters())
print(f"    Total parameters: {total_params:,}")

# Training config
training_args = TrainingArguments(
    output_dir="ai_training/language_model/models",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10,
    save_total_limit=1,
    logging_steps=5,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 6: Train
print(f"\n[*] Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

# Step 7: Save
print(f"\n[*] Saving model...")
output_path = "ai_training/language_model/models/final_model"
os.makedirs(output_path, exist_ok=True)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f"[OK] Model saved to {output_path}")

print("\n" + "="*70)
print("[OK] TRAINING COMPLETE!")
print("="*70)
print(f"\nModel location: {output_path}")
print("\nTo generate text:")
print("  python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('ai_training/language_model/models/final_model'); model = AutoModelForCausalLM.from_pretrained('ai_training/language_model/models/final_model'); inputs = tokenizer('Once', return_tensors='pt'); outputs = model.generate(**inputs, max_length=50); print(tokenizer.decode(outputs[0]))\"")
