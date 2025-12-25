"""
AI Training Setup Script
Creates all necessary directories and installs dependencies
Run: python ai_training_setup.py
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(BASE_DIR, "ai_training")

# Create directory structure
directories = [
    "ai_training",
    "ai_training/image_classifier",
    "ai_training/image_classifier/data/train",
    "ai_training/image_classifier/data/val",
    "ai_training/image_classifier/models",
    "ai_training/language_model",
    "ai_training/language_model/data",
    "ai_training/language_model/models",
    "ai_training/game_ai",
    "ai_training/game_ai/models",
    "ai_training/custom_nn",
    "ai_training/custom_nn/models",
]

print("📁 Creating directory structure...")
for d in directories:
    path = os.path.join(BASE_DIR, d)
    os.makedirs(path, exist_ok=True)
    print(f"  ✓ {d}")

print("\n📦 Installing PyTorch with CUDA...")
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu121"
], check=True)

print("\n📦 Installing other dependencies...")
deps = [
    "numpy", "pandas", "scikit-learn", "matplotlib", "tqdm", "Pillow",
    "transformers", "datasets", "accelerate", "peft",
    "gymnasium", "stable-baselines3",
    "tensorboard", "huggingface_hub"
]
subprocess.run([sys.executable, "-m", "pip", "install"] + deps, check=True)

print("\n✅ Setup complete! You can now run:")
print("  - python ai_training/image_classifier.py")
print("  - python ai_training/language_model.py")
print("  - python ai_training/game_ai.py")
print("  - python ai_training/custom_nn.py")
