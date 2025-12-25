"""
🗣️ Language Model Fine-Tuning with PyTorch
Fine-tune small LLMs on your custom text data with GPU acceleration

USAGE:
1. Add your training text to ai_training/language_model/data/train.txt
   (or use JSON format with {"text": "..."} per line)
2. Run: python language_model.py

Supports: GPT-2, DistilGPT-2, TinyLlama, Phi-2, and more
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, Dataset as HFDataset
import json
from tqdm import tqdm
from datetime import datetime

# ============== CONFIGURATION ==============
CONFIG = {
    "data_dir": "ai_training/language_model/data",
    "model_dir": "ai_training/language_model/models",
    "base_model": "distilgpt2",  # Options: gpt2, distilgpt2, TinyLlama/TinyLlama-1.1B-Chat-v1.0
    "max_length": 512,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch size = batch_size * gradient_accumulation
    "epochs": 3,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "save_steps": 500,
    "logging_steps": 100,
    "use_lora": True,  # Use LoRA for efficient fine-tuning
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
}


class TextDataset(Dataset):
    """Custom dataset for text files"""
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        if file_path.endswith('.json') or file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    text = data.get('text', data.get('content', str(data)))
                    self.examples.append(text)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split into chunks
                chunks = self._chunk_text(content, max_length * 4)
                self.examples = chunks
    
    def _chunk_text(self, text, chunk_size):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size // 2):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk) > 50:  # Skip very short chunks
                chunks.append(chunk)
        return chunks
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class LanguageModelTrainer:
    def __init__(self, config=CONFIG):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model = None
        self.tokenizer = None
        
        os.makedirs(config["data_dir"], exist_ok=True)
        os.makedirs(config["model_dir"], exist_ok=True)
    
    def setup_model(self):
        """Load base model and tokenizer"""
        print(f"\n📥 Loading model: {self.config['base_model']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["base_model"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["base_model"],
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # Apply LoRA if enabled
        if self.config["use_lora"]:
            self._apply_lora()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    def _apply_lora(self):
        """Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config["lora_r"],
                lora_alpha=self.config["lora_alpha"],
                lora_dropout=self.config["lora_dropout"],
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                if "llama" in self.config["base_model"].lower() or "phi" in self.config["base_model"].lower()
                else ["c_attn", "c_proj", "c_fc"]  # For GPT-2
            )
            
            self.model = get_peft_model(self.model, lora_config)
            print("✓ LoRA applied for efficient fine-tuning")
        except ImportError:
            print("⚠️  PEFT not installed. Training full model.")
            print("   Install with: pip install peft")
    
    def prepare_data(self):
        """Load and prepare training data"""
        train_file = os.path.join(self.config["data_dir"], "train.txt")
        
        if not os.path.exists(train_file):
            self._create_sample_data(train_file)
            return None
        
        print(f"\n📊 Loading data from {train_file}")
        dataset = TextDataset(train_file, self.tokenizer, self.config["max_length"])
        print(f"   Training examples: {len(dataset)}")
        
        return dataset
    
    def _create_sample_data(self, file_path):
        """Create sample training data"""
        sample_text = """
This is a sample training file for the language model.
Replace this content with your own training data.

You can add:
- Stories and narratives
- Technical documentation
- Conversations and dialogues
- Any text you want the model to learn from

The model will learn to generate text similar to what you provide.
For best results:
- Use at least 10,000 words of training data
- Keep consistent style and formatting
- Include diverse examples of your target content

Example conversation format:
User: Hello, how are you?
Assistant: I'm doing well, thank you for asking! How can I help you today?

Example story format:
Once upon a time in a digital realm, there lived an AI who loved to learn.
Every day, it would process new information and grow wiser.
The end.
"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        print(f"\n⚠️  No training data found!")
        print(f"   Created sample file: {file_path}")
        print("   Please edit this file with your own training data and run again!")
    
    def train(self):
        """Main training function"""
        self.setup_model()
        dataset = self.prepare_data()
        
        if dataset is None:
            return
        
        # Training arguments
        output_dir = os.path.join(
            self.config["model_dir"], 
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config["epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            warmup_ratio=self.config["warmup_ratio"],
            logging_steps=self.config["logging_steps"],
            save_steps=self.config["save_steps"],
            save_total_limit=3,
            fp16=self.device.type == "cuda",
            report_to="none",
            dataloader_num_workers=0,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        print(f"\n🚀 Starting training...")
        print(f"   Output directory: {output_dir}")
        print(f"   Epochs: {self.config['epochs']}")
        print(f"   Batch size: {self.config['batch_size']} x {self.config['gradient_accumulation_steps']} = {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
        print()
        
        trainer.train()
        
        # Save final model
        final_path = os.path.join(self.config["model_dir"], "final_model")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        print(f"\n✅ Training complete!")
        print(f"   Model saved to: {final_path}")
    
    def generate(self, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
        """Generate text from a prompt"""
        if self.model is None:
            print("⚠️  No model loaded!")
            return None
        
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    def load_model(self, model_path):
        """Load a fine-tuned model"""
        print(f"📂 Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        self.model.eval()
        print("✓ Model loaded!")
    
    def interactive_chat(self):
        """Interactive chat with the model"""
        if self.model is None:
            print("⚠️  No model loaded!")
            return
        
        print("\n💬 Interactive Chat (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            prompt = input("\nYou: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            response = self.generate(prompt, max_new_tokens=150)
            print(f"\nAI: {response}")


class SimpleTransformer(nn.Module):
    """
    A simple transformer-based language model from scratch
    For educational purposes - shows how transformers work
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Causal mask for autoregressive generation
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        x = self.transformer(x, mask=mask)
        x = self.fc_out(x)
        
        return x


if __name__ == "__main__":
    print("=" * 60)
    print("🗣️  LANGUAGE MODEL FINE-TUNING")
    print("=" * 60)
    
    trainer = LanguageModelTrainer()
    
    # Train the model
    trainer.train()
    
    # Example: Load and generate
    # trainer.load_model("ai_training/language_model/models/final_model")
    # print(trainer.generate("Once upon a time"))
    
    # Interactive chat
    # trainer.interactive_chat()
