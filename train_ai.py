"""
🚀 AI TRAINING HUB - Master Launcher
Run all AI training modules from one place

Run: python train_ai.py
"""

import os
import sys
import subprocess

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║            🤖 AI TRAINING HUB - RTX 4050 Edition             ║
╠══════════════════════════════════════════════════════════════╣
║  1. 🖼️  Image Classifier    - CNN with transfer learning     ║
║  2. 🗣️  Language Model      - Fine-tune LLMs (GPT-2, etc)    ║
║  3. 🎮  Game AI             - Reinforcement learning agent   ║
║  4. 🧠  Custom Neural Net   - Build any architecture         ║
║  5. 📦  Install Dependencies                                 ║
║  6. 🔧  Check GPU Status                                     ║
║  0. ❌  Exit                                                  ║
╚══════════════════════════════════════════════════════════════╝
    """)

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        print(f"\n🖥️  PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            props = torch.cuda.get_device_properties(0)
            print(f"   Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"   Compute capability: {props.major}.{props.minor}")
        else:
            print("   ⚠️  No GPU detected! Training will be slow on CPU.")
    except ImportError:
        print("⚠️  PyTorch not installed. Run option 5 first!")

def install_dependencies():
    """Install all required packages"""
    print("\n📦 Installing dependencies...")
    
    # PyTorch with CUDA
    print("\n1/3 Installing PyTorch with CUDA support...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])
    
    # Core packages
    print("\n2/3 Installing core ML packages...")
    core_packages = [
        "numpy", "pandas", "scikit-learn", "matplotlib", 
        "tqdm", "Pillow", "tensorboard"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install"] + core_packages)
    
    # Advanced packages
    print("\n3/3 Installing advanced packages...")
    advanced_packages = [
        "transformers", "datasets", "accelerate", "peft",
        "gymnasium", "stable-baselines3", "huggingface_hub"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install"] + advanced_packages)
    
    print("\n✅ All dependencies installed!")
    check_gpu()

def create_directories():
    """Create required directory structure"""
    base = os.path.dirname(os.path.abspath(__file__))
    dirs = [
        "ai_training",
        "ai_training/image_classifier/data/train",
        "ai_training/image_classifier/data/val",
        "ai_training/image_classifier/models",
        "ai_training/language_model/data",
        "ai_training/language_model/models",
        "ai_training/game_ai/models",
        "ai_training/custom_nn/models",
    ]
    
    for d in dirs:
        path = os.path.join(base, d)
        os.makedirs(path, exist_ok=True)
    
    print("📁 Directory structure created!")

def run_trainer(module):
    """Run a specific trainer module"""
    modules = {
        1: "image_classifier.py",
        2: "language_model.py",
        3: "game_ai.py",
        4: "custom_nn.py",
    }
    
    if module in modules:
        script = modules[module]
        print(f"\n🚀 Launching {script}...\n")
        subprocess.run([sys.executable, script])

def main():
    create_directories()
    
    while True:
        print_banner()
        
        try:
            choice = input("Select option (0-6): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                break
            elif choice == "1":
                run_trainer(1)
            elif choice == "2":
                run_trainer(2)
            elif choice == "3":
                run_trainer(3)
            elif choice == "4":
                run_trainer(4)
            elif choice == "5":
                install_dependencies()
            elif choice == "6":
                check_gpu()
            else:
                print("⚠️  Invalid option. Please select 0-6.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
