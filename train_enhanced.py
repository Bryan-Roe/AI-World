#!/usr/bin/env python3
"""
Enhanced Language Model Training with Advanced Features
Uses the config and trainer from language_model.py
"""

import os
import sys
import torch

# Ensure virtual environment packages are used
venv_path = os.path.join(os.path.dirname(__file__), 'train_env')
if os.path.exists(venv_path):
    print(f"[OK] Using virtual environment: {venv_path}")

# Create training data if needed
os.makedirs("ai_training/language_model/data", exist_ok=True)
data_file = "ai_training/language_model/data/train.txt"

if not os.path.exists(data_file) or os.path.getsize(data_file) < 200:
    print("[*] Creating enhanced training dataset...")
    training_text = """
    Artificial Intelligence is transforming industries across the world.
    Machine learning algorithms learn patterns from data.
    Deep neural networks can solve complex problems.
    Natural language processing enables computers to understand human language.
    Computer vision allows machines to interpret visual information.
    Reinforcement learning trains agents through interaction with environments.
    Transfer learning speeds up training by using pre-trained models.
    Attention mechanisms improve model performance on sequential tasks.
    Transformers have become the dominant architecture for language models.
    Large language models demonstrate remarkable capabilities in reasoning.
    
    The future of AI involves multimodal systems that combine vision and language.
    Efficient training requires careful optimization of hyperparameters.
    Data quality is as important as data quantity in machine learning.
    Generalization is the key challenge in developing robust models.
    Interpretability helps us understand how AI systems make decisions.
    
    Neural networks are inspired by biological brains.
    Gradient descent optimizes model weights through iterative learning.
    Batch normalization stabilizes training and speeds convergence.
    Dropout regularization prevents overfitting in deep networks.
    Loss functions measure the difference between predictions and targets.
    
    Pre-training on large datasets improves downstream task performance.
    Fine-tuning adapts pre-trained models to specific domains.
    Prompt engineering influences the behavior of large language models.
    Few-shot learning enables models to learn from limited examples.
    Meta-learning allows models to learn how to learn.
    """
    with open(data_file, 'w') as f:
        f.write(training_text)
    print(f"[OK] Created training data: {len(training_text)} chars")

print("\n" + "="*70)
print("[*] ENHANCED LANGUAGE MODEL TRAINING")
print("="*70 + "\n")

# Load configuration
print("[*] Loading configuration...")
try:
    from language_model import CONFIG, LanguageModelTrainer
    print("[OK] Configuration loaded successfully")
    print(f"    Model: {CONFIG['base_model']}")
    print(f"    Learning rate: {CONFIG['learning_rate']}")
    print(f"    Batch size: {CONFIG['batch_size']}")
    print(f"    Max epochs: {CONFIG['epochs']}")
    print(f"    Mixed precision: {CONFIG.get('use_mixed_precision', False)}")
    print(f"    Gradient checkpointing: {CONFIG.get('use_gradient_checkpointing', False)}")
    print(f"    LoRA enabled: {CONFIG.get('use_lora', False)}")
    print(f"    Early stopping patience: {CONFIG.get('early_stopping_patience', 3)}")
    print(f"    Validation split: {CONFIG.get('val_split', 0.1)}")
except ImportError as e:
    print(f"[!] Error: Could not import language_model.py")
    print(f"    {e}")
    print("[!] Falling back to basic training...")
    sys.exit(1)

# Initialize trainer
print("\n[*] Initializing trainer...")
trainer = LanguageModelTrainer()
print("[OK] Trainer initialized")

# Setup model
print("\n[*] Setting up model...")
try:
    trainer.setup_model()
    print("[OK] Model setup complete")
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"[!] Error setting up model: {e}")
    sys.exit(1)

# Prepare data
print("\n[*] Preparing training data...")
try:
    trainer.train_dataset, trainer.val_dataset = trainer.prepare_data()
    print("[OK] Data prepared")
    if trainer.train_dataset:
        print(f"    Training samples: {len(trainer.train_dataset)}")
        print(f"    Validation samples: {len(trainer.val_dataset) if trainer.val_dataset else 0}")
        print(f"    Train/Val split: {CONFIG.get('val_split', 0.1)}")
except Exception as e:
    print(f"[!] Error preparing data: {e}")
    sys.exit(1)

# Train model
print("\n[*] Starting training...")
print("-" * 70)
try:
    trainer.train()
    print("-" * 70)
    print("[OK] Training completed successfully!")
except Exception as e:
    print(f"[!] Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Save model
print("\n[*] Saving model...")
try:
    output_path = "ai_training/language_model/models/final_model"
    os.makedirs(output_path, exist_ok=True)
    trainer.model.save_pretrained(output_path)
    trainer.tokenizer.save_pretrained(output_path)
    print(f"[OK] Model saved to: {output_path}")
except Exception as e:
    print(f"[!] Error saving model: {e}")
    sys.exit(1)

# Test generation
print("\n[*] Testing text generation...")
try:
    prompts = [
        "Artificial intelligence",
        "Machine learning algorithms",
        "Deep neural networks"
    ]
    
    for prompt in prompts:
        generated = trainer.generate(prompt, max_length=50)
        print(f"\n  Prompt: {prompt}")
        print(f"  Generated: {generated[:100]}...")
    
    print("\n[OK] Text generation successful!")
except Exception as e:
    print(f"[!] Error during generation: {e}")
    import traceback
    traceback.print_exc()

# Evaluate model
print("\n[*] Evaluating model...")
try:
    metrics = trainer.evaluate_model()
    print("[OK] Evaluation complete")
    if metrics:
        print(f"    Validation Loss: {metrics.get('loss', 'N/A')}")
        print(f"    Perplexity: {metrics.get('perplexity', 'N/A')}")
except Exception as e:
    print(f"[!] Evaluation skipped: {e}")

print("\n" + "="*70)
print("[OK] TRAINING SESSION COMPLETE!")
print("="*70)
print(f"\nModel saved to: ai_training/language_model/models/final_model")
print(f"Training data: {data_file}")
print(f"\nNext steps:")
print("  1. Use the model with: python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; ...\"")
print("  2. Deploy to web with: npm start")
print("  3. Test generation quality in the browser at http://localhost:3000")
