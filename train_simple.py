#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple LLM Training Script
Trains language model with minimal dependencies
"""

import sys
import subprocess
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

# Check and install dependencies
required_packages = [
    'torch',
    'transformers',
    'datasets',
    'peft',
    'tqdm'
]

print("[*] Checking dependencies...")
missing = []

for package in required_packages:
    try:
        __import__(package)
        print(f"    [OK] {package}")
    except ImportError:
        print(f"    [!] {package} - installing...")
        missing.append(package)

if missing:
    print(f"\n[*] Installing {len(missing)} package(s)...")
    for pkg in missing:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
            print(f"    [OK] Installed {pkg}")
        except Exception as e:
            print(f"    [!] Failed to install {pkg}: {e}")

# Now run training
print("\n" + "="*60)
print("[*] LANGUAGE MODEL TRAINING")
print("="*60)

try:
    from language_model import LanguageModelTrainer, CONFIG
    
    print("\n[*] Configuration:")
    print(f"    Model: {CONFIG['base_model']}")
    print(f"    Data: {CONFIG['data_dir']}")
    print(f"    Max length: {CONFIG['max_length']}")
    print(f"    Epochs: {CONFIG['epochs']}")
    print(f"    Batch size: {CONFIG['batch_size']}")
    print(f"    Mixed precision: {CONFIG['use_mixed_precision']}")
    print(f"    LoRA: {CONFIG['use_lora']}")
    print()
    
    trainer = LanguageModelTrainer(CONFIG)
    trainer.train()
    
except ImportError as e:
    print(f"\n[!] Import error: {e}")
    print("\nTrying direct import...")
    exec(open('language_model.py').read())
except Exception as e:
    print(f"\n[!] Error: {e}")
    import traceback
    traceback.print_exc()
