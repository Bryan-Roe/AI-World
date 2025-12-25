#!/usr/bin/env python3
"""
Production Language Model Training - Final Version
Complete training with all optimizations
"""

import os
import sys
import torch

# Setup paths
os.makedirs("ai_training/language_model/data", exist_ok=True)
os.makedirs("ai_training/language_model/models", exist_ok=True)

# Create training data if needed
data_file = "ai_training/language_model/data/train.txt"
if not os.path.exists(data_file) or os.path.getsize(data_file) < 200:
    print("[*] Creating training dataset...")
    training_text = """
Artificial Intelligence is transforming industries across the world.
Machine learning algorithms learn patterns from data with high accuracy.
Deep neural networks can solve complex optimization and classification problems.
Natural language processing enables computers to understand human language effectively.
Computer vision allows machines to interpret visual information from images and videos.
Reinforcement learning trains agents through interaction with environments.
Transfer learning speeds up training by using pre-trained models effectively.
Attention mechanisms improve model performance on sequential processing tasks.
Transformers have become the dominant architecture for language models globally.
Large language models demonstrate remarkable capabilities in reasoning tasks.

The future of artificial intelligence involves multimodal systems that combine vision and language.
Efficient training requires careful optimization of many hyperparameters systematically.
Data quality is as important as data quantity in machine learning systems.
Generalization is the key challenge in developing robust machine learning models.
Interpretability helps us understand how artificial intelligence systems make decisions.

Neural networks are inspired by biological brains and their organization.
Gradient descent optimizes model weights through iterative learning processes.
Batch normalization stabilizes training and speeds convergence significantly.
Dropout regularization prevents overfitting in deep neural network architectures.
Loss functions measure the difference between predictions and target values.

Pre-training on large datasets improves downstream task performance dramatically.
Fine-tuning adapts pre-trained models to specific domains and applications.
Prompt engineering influences the behavior of large language models effectively.
Few-shot learning enables models to learn from limited training examples.
Meta-learning allows models to learn how to learn new tasks quickly.

    """
    with open(data_file, 'w') as f:
        f.write(training_text)
    print(f"[OK] Created {len(training_text)} chars of training data")

print("\n" + "="*70)
print("[*] PRODUCTION LANGUAGE MODEL TRAINING")
print("="*70 + "\n")

# Install dependencies
print("[*] Checking dependencies...")
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    print("[OK] All transformers dependencies available")
except ImportError as e:
    print(f"[!] Installing missing dependencies...")
    os.system(f"{sys.executable} -m pip install transformers datasets -q")
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset

# Load configuration
print("[*] Loading training configuration...")
CONFIG = {
    "base_model": "distilgpt2",
    "max_length": 256,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "epochs": 2,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "use_mixed_precision": True,
    "use_gradient_checkpointing": True,
}

print(f"[OK] Configuration:")
print(f"    Model: {CONFIG['base_model']}")
print(f"    Learning rate: {CONFIG['learning_rate']}")
print(f"    Batch size: {CONFIG['batch_size']} x {CONFIG['gradient_accumulation']}")
print(f"    Epochs: {CONFIG['epochs']}")
print(f"    Mixed precision: {CONFIG['use_mixed_precision']}")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[*] Using device: {device}")
if device.type == "cuda":
    print(f"    GPU: {torch.cuda.get_device_name(0)}")

# Load model and tokenizer
print(f"\n[*] Loading model: {CONFIG['base_model']}...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["base_model"],
    torch_dtype=torch.float32,
)
model.to(device)

# Enable optimizations
if CONFIG["use_gradient_checkpointing"]:
    model.gradient_checkpointing_enable()

print(f"[OK] Model loaded")
total_params = sum(p.numel() for p in model.parameters())
print(f"    Total parameters: {total_params:,}")

# Load and tokenize data
print(f"\n[*] Loading training data from {data_file}...")
with open(data_file, 'r') as f:
    text = f.read()

print(f"[OK] Loaded {len(text)} characters")

# Create dataset
print("[*] Tokenizing data...")
tokens = tokenizer(
    text,
    truncation=True,
    max_length=CONFIG["max_length"],
    padding=True,
    return_tensors="pt"
)

# Create torch dataset
examples = []
for i in range(tokens['input_ids'].shape[0]):
    examples.append({
        'input_ids': tokens['input_ids'][i].tolist(),
        'labels': tokens['input_ids'][i].tolist()
    })

dataset = Dataset.from_list(examples)
print(f"[OK] Created dataset with {len(dataset)} examples")

if len(dataset) == 0:
    print("[!] Error: No training examples!")
    sys.exit(1)

# Train
print("\n[*] Setting up trainer...")
training_args = TrainingArguments(
    output_dir="ai_training/language_model/models/final",
    overwrite_output_dir=True,
    num_train_epochs=CONFIG["epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation"],
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    report_to="none",
    fp16=CONFIG["use_mixed_precision"] and device.type == "cuda",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("[OK] Trainer configured")
print("-" * 70)
print("[*] Starting training...")
print("-" * 70 + "\n")

try:
    trainer.train()
except KeyboardInterrupt:
    print("\n[!] Training interrupted by user")
except Exception as e:
    print(f"\n[!] Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "-" * 70)
print("[OK] Training completed!")
print("-" * 70)

# Save model
print("\n[*] Saving model...")
output_dir = "ai_training/language_model/models/final_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"[OK] Model saved to: {output_dir}")

# Test generation
print("\n[*] Testing text generation...")
test_prompts = [
    "Artificial intelligence",
    "Machine learning",
    "Neural networks"
]

model.eval()
with torch.no_grad():
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        outputs = model.generate(
            **inputs,
            max_length=60,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  Prompt: {prompt}")
        print(f"  Generated: {generated[:100]}...")

print("\n" + "="*70)
print("[OK] TRAINING COMPLETE!")
print("="*70)
print(f"\nModel location: {output_dir}")
print(f"Training data: {data_file}")
print("\nNext steps:")
print("  1. Deploy to web: npm start")
print("  2. Test in browser: http://localhost:3000")
print("  3. Use for generation: See API documentation")
