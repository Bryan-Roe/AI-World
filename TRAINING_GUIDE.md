# Language Model Training Guide

## Quick Start

### 1. Install Dependencies

```bash
# Install core ML packages (this may take 10-20 minutes)
pip install torch transformers datasets peft tqdm --quiet

# Optional: For faster training with GPU (if you have NVIDIA GPU)
pip install cuda-toolkit cupy --quiet
```

### 2. Prepare Training Data

Place your training text in:
```
ai_training/language_model/data/train.txt
```

Example training data:
```
Once upon a time, there was a magical forest...

The ancient library held secrets of forgotten civilizations...

Welcome to the AI training tutorial. Today we'll learn about language models...
```

**Recommendations:**
- At least 10,000 words (5,000+ examples)
- Multiple topics for diversity
- Consistent formatting
- Mix of sentence lengths

### 3. Run Training

```bash
# Method 1: Direct training
python language_model.py

# Method 2: With dependency checking
python train_simple.py

# Method 3: Interactive
python -c "from language_model import LanguageModelTrainer; LanguageModelTrainer().train()"
```

### 4. Monitor Training

You should see output like:
```
🖥️  Using device: cuda
   GPU: NVIDIA GeForce RTX 3090
   Memory: 24.0 GB
...
✓ LoRA applied for efficient fine-tuning
   Total parameters: 82,559,488
   Trainable parameters: 1,638,400
   Trainable %: 1.99%

📊 Loading data from ai_training/language_model/data/train.txt
   Total examples: 500
   Training examples: 450
   Validation examples: 50

🚀 Starting training with validation...
   Output directory: ai_training/language_model/models/run_20251225_143022
   Epochs: 3
   Batch size: 4 x 4 = 16
   Learning rate scheduler: cosine
   Early stopping patience: 3
```

## Training Configuration

Edit `CONFIG` in `language_model.py` to customize:

```python
CONFIG = {
    # Data
    "data_dir": "ai_training/language_model/data",
    "model_dir": "ai_training/language_model/models",
    
    # Model choice
    "base_model": "distilgpt2",  # Fast & efficient
    # Alternatives:
    # "gpt2" - Larger, better quality
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0" - Advanced
    # "microsoft/phi-2" - High quality
    
    # Sequence & batch
    "max_length": 512,              # Token length per example
    "batch_size": 4,                # Examples per batch
    "gradient_accumulation_steps": 4,  # Effective batch: 4*4=16
    
    # Training
    "epochs": 3,                    # Full training passes
    "learning_rate": 5e-5,          # Learning rate
    "warmup_ratio": 0.1,            # Warmup period
    "weight_decay": 0.01,           # L2 regularization
    
    # Optimization flags
    "use_mixed_precision": True,    # Float16 on GPU - 2x faster
    "use_gradient_checkpointing": True,  # Save memory (50% less)
    "use_cosine_scheduler": True,   # Better convergence
    "use_data_augmentation": True,  # Improve generalization
    "use_lora": True,               # LoRA - 95% fewer parameters
    "lora_r": 16,                   # LoRA rank
    "lora_alpha": 32,               # LoRA scale
    
    # Monitoring
    "val_split": 0.1,               # 10% for validation
    "early_stopping_patience": 3,   # Stop after 3 epochs no improvement
    
    # Checkpointing
    "save_steps": 500,              # Save every 500 steps
    "logging_steps": 100,           # Log every 100 steps
}
```

## Performance Tips

### For Faster Training
```python
CONFIG = {
    "use_gradient_checkpointing": False,  # Disable memory saving
    "batch_size": 8,                      # Larger batches
    "epochs": 1,                          # Fewer epochs
    "use_mixed_precision": True,          # Keep this enabled
}
```

### For Better Quality
```python
CONFIG = {
    "base_model": "gpt2",                 # Larger model
    "epochs": 10,                         # More training
    "num_training_samples": 1000,         # More data
    "learning_rate": 1e-4,                # Slightly higher LR
    "max_length": 1024,                   # Longer sequences
}
```

### For Limited GPU Memory (< 4GB)
```python
CONFIG = {
    "use_gradient_checkpointing": True,   # Essential
    "use_mixed_precision": True,          # Essential
    "batch_size": 1,                      # Minimum
    "gradient_accumulation_steps": 8,     # Simulate batch_size: 8
    "max_length": 256,                    # Shorter sequences
    "use_lora": True,                     # Essential
}
```

### For CPU-Only Training
```python
CONFIG = {
    "use_mixed_precision": False,         # Not supported on CPU
    "batch_size": 2,                      # Very small
    "gradient_accumulation_steps": 8,     # Compensate
    "epochs": 1,                          # Training is slow
    "base_model": "distilgpt2",           # Smallest model
}
```

## Training Process

### Phase 1: Data Loading (5-10 seconds)
- Reads training data
- Tokenizes text
- Creates data loaders

### Phase 2: Model Loading (30 seconds - 2 minutes)
- Downloads base model
- Applies LoRA/optimizations
- Shows parameter count

### Phase 3: Training Loop (varies by data size)
- **Epoch 1**: ~2-10 minutes per 1000 examples
- **Epoch 2**: Same duration
- **Epoch 3**: Same duration
- Validation happens between epochs
- Early stopping may trigger early

### Phase 4: Saving (30 seconds - 1 minute)
- Saves best model
- Saves tokenizer
- Saves metrics

**Total Time:** 10-60 minutes depending on data size and hardware

## Output Files

After training, check:
```
ai_training/language_model/models/final_model/
├── config.json              # Model configuration
├── pytorch_model.bin        # Model weights (~350 MB for distilgpt2)
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json    # Tokenizer config
├── metrics.json             # Final metrics
│   ├── eval_loss: 3.2415
│   └── eval_runtime: 12.5
└── adapter_config.json      # LoRA config (if used)
```

## Generate Text After Training

```python
from language_model import LanguageModelTrainer

# Load trained model
trainer = LanguageModelTrainer()
trainer.load_model("ai_training/language_model/models/final_model")

# Generate text
prompt = "The future of AI is"
text = trainer.generate(
    prompt,
    max_new_tokens=150,
    temperature=0.7,      # Lower = more focused
    top_p=0.9            # Lower = more conservative
)
print(text)
```

## Interactive Chat

```python
from language_model import LanguageModelTrainer

trainer = LanguageModelTrainer()
trainer.load_model("ai_training/language_model/models/final_model")
trainer.interactive_chat()
```

Then chat like:
```
You: Hello, how are you?
AI: [Generated response]

You: Tell me about the future
AI: [Generated response]

You: quit
```

## Evaluate Model Quality

```python
from language_model import LanguageModelTrainer, TextDataset

trainer = LanguageModelTrainer()
trainer.load_model("ai_training/language_model/models/final_model")

# Load evaluation data
eval_dataset = TextDataset(
    "ai_training/language_model/data/train.txt",
    trainer.tokenizer,
    512
)

# Get metrics
metrics = trainer.evaluate_model(eval_dataset)
# Shows:
# - Average Loss
# - Perplexity (lower is better)
```

**Quality Metrics:**
- **Loss < 3.0**: Good training
- **Perplexity < 20**: Good quality
- **Perplexity < 10**: Excellent quality

## Troubleshooting

### CUDA Out of Memory
```python
CONFIG = {
    "use_gradient_checkpointing": True,
    "batch_size": 1,
    "max_length": 256,
}
```

### Model Generates Repetitive Text
```python
# Adjust generation parameters
trainer.generate(
    prompt,
    max_new_tokens=100,
    temperature=0.5,      # Lower temperature
    top_p=0.7,           # Lower top_p
    repetition_penalty=1.5  # Higher penalty
)
```

### Training Loss Not Decreasing
1. Check training data quality
2. Increase learning_rate: 1e-4
3. Try different base_model: gpt2
4. Increase epochs: 5-10

### "No training data found"
1. Create file: `ai_training/language_model/data/train.txt`
2. Add at least 100 words
3. Run training again

## Advanced Features

### Custom Prompt Engineering
```python
# Fine-tune on specific format
training_data = """
User: What is AI?
Assistant: AI stands for Artificial Intelligence...

User: How does machine learning work?
Assistant: Machine learning uses data to train models...
"""
```

### Multi-Task Training
Train on multiple domains:
```python
# Mix different content types in training.txt
# Stories, Q&A, technical docs, etc.
```

### Transfer Learning
Start from larger model:
```python
CONFIG = {
    "base_model": "microsoft/phi-2",  # High quality base
    "epochs": 3,
    "learning_rate": 1e-5,  # Lower LR for fine-tuning
}
```

## Next Steps

1. **Prepare Data**: Create high-quality training dataset
2. **Configure**: Adjust CONFIG for your hardware
3. **Train**: Run `python language_model.py`
4. **Evaluate**: Check metrics and generation quality
5. **Deploy**: Use trained model in your app
6. **Iterate**: Improve with more/better data

## Need Help?

Check these files:
- [LANGUAGE_MODEL_ENHANCEMENTS.md](LANGUAGE_MODEL_ENHANCEMENTS.md) - Technical details
- [language_model.py](language_model.py) - Source code
- [README.md](README.md) - Project overview
