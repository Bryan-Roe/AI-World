"""
🌍 World Generator - Train LLM to Generate 3D Game Worlds
Fine-tune a language model to create procedural game environments

USAGE:
1. Run: python world_generator_train.py
2. Generates training data and trains the model
3. Use trained model to generate worlds in game.js

The model learns to generate JSON world descriptions with:
- Terrain (planes, hills, valleys)
- Objects (cubes, spheres, trees, buildings)
- Lighting (ambient, directional, point lights)
- Colors, positions, scales
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
    DataCollatorForLanguageModeling
)
import json
from datetime import datetime
from tqdm import tqdm
import random

# ============== CONFIGURATION ==============
CONFIG = {
    "data_dir": "ai_training/world_generator/data",
    "model_dir": "ai_training/world_generator/models",
    "base_model": "distilgpt2",  # Fast and efficient for world generation
    "max_length": 1024,  # Longer context for complete world descriptions
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "epochs": 5,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "save_steps": 250,
    "logging_steps": 50,
    "num_training_samples": 500,  # Generate synthetic training data
}

# World generation templates and components
TERRAIN_TYPES = ["flat", "hilly", "mountainous", "valley", "plateau", "canyon"]
BIOMES = ["desert", "forest", "ocean", "arctic", "grassland", "volcanic", "alien"]
OBJECT_TYPES = ["cube", "sphere", "cylinder", "cone", "torus", "pyramid"]
STRUCTURES = ["house", "tower", "bridge", "wall", "arch", "pillar"]
COLORS = {
    "terrain": ["0x2d5016", "0x8b7355", "0x4a4a4a", "0xf4a460", "0x90ee90"],
    "objects": ["0xff0000", "0x00ff00", "0x0000ff", "0xffff00", "0xff00ff", "0x00ffff"],
    "sky": ["0x87ceeb", "0x1a1a2e", "0xff4500", "0x800080", "0x191970"]
}


def create_directories():
    """Create necessary directories"""
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    print(f"✅ Created directories: {CONFIG['data_dir']}, {CONFIG['model_dir']}")


def generate_world_description():
    """Generate a synthetic world description"""
    biome = random.choice(BIOMES)
    terrain = random.choice(TERRAIN_TYPES)
    
    # Base world properties
    world = {
        "name": f"{biome.capitalize()} {terrain.capitalize()} World",
        "biome": biome,
        "terrain": terrain,
        "sky_color": random.choice(COLORS["sky"]),
        "fog_density": round(random.uniform(0.001, 0.015), 4),
        "terrain_objects": [],
        "structures": [],
        "lights": []
    }
    
    # Generate terrain (ground plane + features)
    num_terrain_features = random.randint(3, 8)
    for _ in range(num_terrain_features):
        terrain_obj = {
            "type": random.choice(["plane", "cylinder", "sphere"]),
            "position": [
                round(random.uniform(-100, 100), 2),
                round(random.uniform(-5, 0), 2),
                round(random.uniform(-100, 100), 2)
            ],
            "scale": [
                round(random.uniform(10, 50), 2),
                round(random.uniform(1, 5), 2),
                round(random.uniform(10, 50), 2)
            ],
            "color": random.choice(COLORS["terrain"]),
            "receives_shadow": True
        }
        world["terrain_objects"].append(terrain_obj)
    
    # Generate structures/objects
    num_objects = random.randint(5, 15)
    for _ in range(num_objects):
        obj_type = random.choice(OBJECT_TYPES + STRUCTURES)
        obj = {
            "type": obj_type,
            "position": [
                round(random.uniform(-80, 80), 2),
                round(random.uniform(0, 30), 2),
                round(random.uniform(-80, 80), 2)
            ],
            "scale": [
                round(random.uniform(1, 8), 2),
                round(random.uniform(1, 8), 2),
                round(random.uniform(1, 8), 2)
            ],
            "color": random.choice(COLORS["objects"]),
            "rotation": round(random.uniform(0, 360), 2),
            "casts_shadow": True
        }
        world["structures"].append(obj)
    
    # Generate lights
    num_lights = random.randint(2, 5)
    light_types = ["ambient", "directional", "point", "spot"]
    for _ in range(num_lights):
        light_type = random.choice(light_types)
        light = {
            "type": light_type,
            "color": f"0x{random.randint(0xaaaaaa, 0xffffff):06x}",
            "intensity": round(random.uniform(0.3, 1.5), 2)
        }
        if light_type in ["point", "spot"]:
            light["position"] = [
                round(random.uniform(-50, 50), 2),
                round(random.uniform(10, 40), 2),
                round(random.uniform(-50, 50), 2)
            ]
        if light_type in ["directional", "spot"]:
            light["direction"] = [
                round(random.uniform(-1, 1), 2),
                round(random.uniform(-1, 0), 2),
                round(random.uniform(-1, 1), 2)
            ]
        world["lights"].append(light)
    
    return world


def create_training_prompt(world_desc):
    """Convert world description to training format"""
    # Create natural language prompt + JSON response
    prompt = f"Generate a {world_desc['biome']} world with {world_desc['terrain']} terrain:"
    response = json.dumps(world_desc, indent=2)
    return f"{prompt}\n\n{response}"


def generate_training_data():
    """Generate synthetic training data for world generation"""
    print(f"🔄 Generating {CONFIG['num_training_samples']} training samples...")
    
    training_file = os.path.join(CONFIG["data_dir"], "world_training.jsonl")
    
    with open(training_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(CONFIG["num_training_samples"])):
            world = generate_world_description()
            training_text = create_training_prompt(world)
            
            # Save as JSONL
            json.dump({"text": training_text}, f)
            f.write('\n')
    
    print(f"✅ Generated training data: {training_file}")
    return training_file


class WorldDataset(Dataset):
    """Dataset for world generation training"""
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"📖 Loading training data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                text = data.get('text', '')
                if text:
                    self.examples.append(text)
        
        print(f"✅ Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }


def train_world_generator():
    """Train the world generation model"""
    print("\n🌍 WORLD GENERATOR TRAINING")
    print("=" * 60)
    
    # Setup
    create_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    # Generate training data
    training_file = generate_training_data()
    
    # Load tokenizer and model
    print(f"\n📦 Loading model: {CONFIG['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"])
    model.to(device)
    
    # Create dataset
    dataset = WorldDataset(training_file, tokenizer, CONFIG["max_length"])
    
    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG["model_dir"], f"world_gen_{timestamp}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        save_steps=CONFIG["save_steps"],
        logging_steps=CONFIG["logging_steps"],
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n🚀 Starting training for {CONFIG['epochs']} epochs...")
    print(f"📊 Batch size: {CONFIG['batch_size']} × {CONFIG['gradient_accumulation_steps']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(CONFIG["model_dir"], "world_generator_final")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {final_model_path}")
    
    return model, tokenizer, final_model_path


def test_generation(model, tokenizer, num_samples=3):
    """Test the trained model"""
    print(f"\n🧪 Generating {num_samples} test worlds...")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_prompts = [
        "Generate a forest world with hilly terrain:",
        "Generate a desert world with flat terrain:",
        "Generate an alien world with mountainous terrain:"
    ]
    
    for i, prompt in enumerate(test_prompts[:num_samples], 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {prompt}")
        print('='*60)
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=800,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        
        # Try to parse JSON
        try:
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                world_json = generated_text[json_start:json_end]
                world = json.loads(world_json)
                print(f"\n✅ Valid world generated with {len(world.get('structures', []))} structures")
        except:
            print("\n⚠️ Generated text is not valid JSON (needs more training)")


def create_inference_script():
    """Create a standalone script for world generation"""
    script_path = "generate_world.py"
    
    script_content = '''"""
Quick World Generator - Use trained model to generate worlds
Run: python generate_world.py "forest hilly"
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys

MODEL_PATH = "ai_training/world_generator/models/world_generator_final"

def generate_world(prompt_text):
    """Generate a world from text prompt"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    # Format prompt
    prompt = f"Generate a {prompt_text} world:"
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=800,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON
    json_start = generated.find('{')
    json_end = generated.rfind('}') + 1
    if json_start != -1 and json_end > json_start:
        world_json = generated[json_start:json_end]
        try:
            world = json.loads(world_json)
            print("\\n✅ Generated World:")
            print(json.dumps(world, indent=2))
            
            # Save to file
            with open("generated_world.json", "w") as f:
                json.dump(world, f, indent=2)
            print("\\n📁 Saved to: generated_world.json")
            
            return world
        except:
            print("⚠️ Could not parse JSON")
            print(generated)
    else:
        print("⚠️ No JSON found in output")
        print(generated)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "forest hilly"
    
    generate_world(prompt)
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created inference script: {script_path}")


def main():
    """Main training pipeline"""
    print("""
    🌍 WORLD GENERATOR TRAINING
    ===========================
    This will train a language model to generate 3D game worlds
    
    Process:
    1. Generate synthetic training data (world descriptions)
    2. Fine-tune language model on this data
    3. Test generation capabilities
    4. Create inference scripts
    
    """)
    
    # Train
    model, tokenizer, model_path = train_world_generator()
    
    # Test
    test_generation(model, tokenizer, num_samples=3)
    
    # Create helper scripts
    create_inference_script()
    
    print(f"""
    
    ✅ TRAINING COMPLETE!
    =====================
    
    📁 Model Location: {model_path}
    
    🚀 Next Steps:
    
    1. Test generation:
       python generate_world.py "alien mountainous"
    
    2. Integrate with game:
       - Load generated_world.json in game.js
       - Parse and render the world description
    
    3. Improve model:
       - Adjust CONFIG['num_training_samples'] for more data
       - Increase CONFIG['epochs'] for better quality
       - Use larger base model (gpt2 instead of distilgpt2)
    
    Happy world generating! 🌍✨
    """)


if __name__ == "__main__":
    main()
