#!/usr/bin/env python3
"""
Minimal Language Model Training Script
Handles dependency installation smoothly
"""

import sys
import os

# Set Python encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

def install_dependencies():
    """Install required packages silently"""
    import subprocess
    packages = ['torch', 'transformers', 'datasets', 'peft', 'tqdm']
    
    print("[*] Installing dependencies (this may take 5-15 minutes)...")
    print("[*] Downloading PyTorch, transformers, and other packages...")
    print()
    
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"    Installing {pkg}...")
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', pkg, '--quiet'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"    [OK] {pkg}")
            except Exception as e:
                print(f"    [!] Could not install {pkg}: {e}")
                return False
    
    print("\n[OK] All dependencies installed\n")
    return True

if __name__ == "__main__":
    # Install dependencies
    if not install_dependencies():
        print("[!] Failed to install dependencies")
        sys.exit(1)
    
    # Import training module
    try:
        from language_model import LanguageModelTrainer, CONFIG
        
        print("="*70)
        print("[*] LANGUAGE MODEL TRAINING WITH ENHANCEMENTS")
        print("="*70)
        print()
        print("[*] Configuration:")
        print(f"    Base Model:          {CONFIG['base_model']}")
        print(f"    Data Directory:      {CONFIG['data_dir']}")
        print(f"    Max Sequence Length: {CONFIG['max_length']}")
        print(f"    Training Epochs:     {CONFIG['epochs']}")
        print(f"    Batch Size:          {CONFIG['batch_size']}")
        print(f"    Gradient Accum:      {CONFIG['gradient_accumulation_steps']}")
        print(f"    Learning Rate:       {CONFIG['learning_rate']}")
        print()
        print("[*] Enhanced Features:")
        print(f"    Mixed Precision:     {CONFIG['use_mixed_precision']}")
        print(f"    Gradient Checkpt:    {CONFIG['use_gradient_checkpointing']}")
        print(f"    LoRA Fine-tuning:    {CONFIG['use_lora']}")
        print(f"    Data Augmentation:   {CONFIG['use_data_augmentation']}")
        print(f"    Cosine Scheduler:    {CONFIG['use_cosine_scheduler']}")
        print(f"    Validation Split:    {CONFIG['val_split']:.0%}")
        print(f"    Early Stopping:      {CONFIG['early_stopping_patience']} epochs")
        print()
        print("-"*70)
        print()
        
        # Run training
        trainer = LanguageModelTrainer(CONFIG)
        trainer.train()
        
        print()
        print("="*70)
        print("[OK] TRAINING COMPLETE!")
        print("="*70)
        print()
        print("Model saved to: ai_training/language_model/models/final_model")
        print()
        print("Next steps:")
        print("1. Generate text: python -c \"from language_model import LanguageModelTrainer; trainer = LanguageModelTrainer(); trainer.load_model('ai_training/language_model/models/final_model'); print(trainer.generate('Once upon a time'))\"")
        print("2. Interactive chat: trainer.interactive_chat()")
        print("3. Evaluate: trainer.evaluate_model(eval_dataset)")
        print()
        
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
