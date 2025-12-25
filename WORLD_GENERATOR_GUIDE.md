# 🌍 World Generator Guide

## Overview

Train a local language model (LLM) to generate procedural 3D game worlds with custom terrain, objects, lighting, and atmosphere.

## Quick Start

### 1. Train the Model (One-Time Setup)

```bash
# Install dependencies (if not already done)
pip install transformers datasets torch accelerate

# Train the world generator (5-10 minutes on CPU, 2-3 minutes on GPU)
python world_generator_train.py
```

**Training Progress:**
- Generates 500 synthetic world descriptions
- Fine-tunes DistilGPT-2 for 5 epochs
- Saves model to `ai_training/world_generator/models/world_generator_final`
- Creates `generate_world.py` for standalone generation

### 2. Generate Worlds

#### Option A: Web Interface (Recommended)
1. Start the server: `npm run dev`
2. Open: http://localhost:3000/world_generator.html
3. Select biome and terrain type
4. Click "Generate World"
5. View in 3D or copy JSON

#### Option B: Command Line
```bash
# Generate a specific world
python generate_world.py "alien mountainous"

# Output saved to: generated_world.json
```

#### Option C: Direct API
```bash
curl -X POST http://localhost:3000/api/generate-world \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Generate a forest world with hilly terrain:"}'
```

## World Structure

Generated worlds follow this JSON schema:

```json
{
  "name": "Forest Hilly World",
  "biome": "forest",
  "terrain": "hilly",
  "sky_color": "0x87ceeb",
  "fog_density": 0.008,
  "terrain_objects": [
    {
      "type": "plane",
      "position": [0, -2, 0],
      "scale": [100, 1, 100],
      "color": "0x2d5016",
      "receives_shadow": true
    }
  ],
  "structures": [
    {
      "type": "cube",
      "position": [10, 5, -20],
      "scale": [3, 3, 3],
      "color": "0xff0000",
      "rotation": 45,
      "casts_shadow": true
    }
  ],
  "lights": [
    {
      "type": "directional",
      "color": "0xffffff",
      "intensity": 1.0,
      "direction": [-1, -1, 0]
    }
  ]
}
```

## Configuration Options

### Training Configuration (`world_generator_train.py`)

```python
CONFIG = {
    "num_training_samples": 500,  # More = better quality, longer training
    "epochs": 5,                   # More = better quality, longer training
    "batch_size": 2,               # Adjust based on RAM/VRAM
    "max_length": 1024,            # Max tokens for world description
    "base_model": "distilgpt2",    # or "gpt2" for larger model
}
```

### Biome Types
- **forest** 🌲 - Green terrain, tree-like structures
- **desert** 🏜️ - Sandy colors, sparse objects
- **ocean** 🌊 - Blue tones, aquatic theme
- **arctic** ❄️ - White/blue, icy structures
- **grassland** 🌾 - Rolling plains
- **volcanic** 🌋 - Red/orange, dramatic lighting
- **alien** 👽 - Exotic colors and shapes

### Terrain Types
- **flat** - Level ground
- **hilly** - Rolling elevation
- **mountainous** - Steep peaks
- **valley** - Low between high areas
- **plateau** - Elevated flat areas
- **canyon** - Deep cuts in terrain

## Integration with 3D Game

### Load Generated World in game.js

```javascript
// In game.js, add this method to WorldGenerator class
async loadGeneratedWorld(worldData) {
  // Clear existing world
  this.clearWorld();
  
  // Apply sky settings
  this.scene.background = new THREE.Color(worldData.sky_color);
  this.scene.fog.density = worldData.fog_density;
  
  // Create terrain
  worldData.terrain_objects.forEach(obj => {
    let geometry;
    switch(obj.type) {
      case 'plane':
        geometry = new THREE.PlaneGeometry(obj.scale[0], obj.scale[2]);
        break;
      case 'sphere':
        geometry = new THREE.SphereGeometry(obj.scale[0]);
        break;
      // ... other types
    }
    
    const material = new THREE.MeshStandardMaterial({ 
      color: obj.color 
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...obj.position);
    mesh.receiveShadow = obj.receives_shadow;
    this.scene.add(mesh);
  });
  
  // Create structures (same pattern)
  worldData.structures.forEach(obj => {
    // ... create and position objects
  });
  
  // Create lights
  worldData.lights.forEach(light => {
    let lightObj;
    switch(light.type) {
      case 'directional':
        lightObj = new THREE.DirectionalLight(
          light.color, 
          light.intensity
        );
        if (light.direction) {
          lightObj.position.set(...light.direction);
        }
        break;
      // ... other light types
    }
    this.scene.add(lightObj);
  });
}
```

## Advanced Usage

### Custom Training Data

Create your own world descriptions:

```python
# Add to world_generator_train.py
custom_worlds = [
    {
        "prompt": "Generate a mystical floating islands world:",
        "world": {
            "name": "Floating Islands",
            "biome": "alien",
            "terrain": "mountainous",
            # ... full world description
        }
    }
]

# Append to training data
with open(training_file, 'a') as f:
    for item in custom_worlds:
        json.dump({"text": create_training_prompt(item["world"])}, f)
        f.write('\n')
```

### Improve Generation Quality

1. **More Training Data:**
   ```python
   CONFIG["num_training_samples"] = 1000  # Default: 500
   ```

2. **More Training Epochs:**
   ```python
   CONFIG["epochs"] = 10  # Default: 5
   ```

3. **Larger Model:**
   ```python
   CONFIG["base_model"] = "gpt2"  # Default: distilgpt2
   ```

4. **Use GPU:**
   - Install CUDA-enabled PyTorch
   - Training will be 10-20x faster

### Adjust Generation Parameters

In `generate_world.py`, modify the generation settings:

```python
outputs = model.generate(
    **inputs,
    max_length=800,        # Longer = more detailed
    temperature=0.8,       # Higher = more creative (0.1-1.5)
    top_p=0.95,           # Nucleus sampling (0.9-0.99)
    num_return_sequences=1,
    do_sample=True
)
```

## Troubleshooting

### Model Not Found Error
```
Error: World generator model not found
```
**Solution:** Run `python world_generator_train.py` first

### Out of Memory During Training
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `CONFIG["batch_size"]` to 1
- Use `"base_model": "distilgpt2"` (smaller)
- Train on CPU (slower but no memory limit)

### Invalid JSON Generated
```
⚠️ Generated text is not valid JSON
```
**Solutions:**
- Train for more epochs (model needs more learning)
- Increase `num_training_samples` (more diverse data)
- Adjust `temperature` (lower = more consistent)

### Slow Training
Training takes 43+ seconds per iteration on CPU.

**Solutions:**
- **Use GPU:** 10-20x faster with CUDA
- **Reduce data:** Lower `num_training_samples`
- **Smaller model:** Use `distilgpt2` (default)
- **Run overnight:** Let it complete (5-10 hours on CPU)

## Performance Tips

### Training Performance
- **CPU:** ~2-3 hours for 500 samples, 5 epochs
- **GPU (RTX 3060):** ~5-10 minutes
- **GPU (RTX 4090):** ~2-3 minutes

### Generation Performance
- **CPU:** 2-5 seconds per world
- **GPU:** 0.5-1 second per world

### Optimize for Production
```python
# In generate_world.py, add caching
import functools

@functools.lru_cache(maxsize=32)
def load_model():
    """Cache loaded model"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    return model, tokenizer

# Reuse model across generations
model, tokenizer = load_model()
```

## Architecture Details

### Model
- **Base:** DistilGPT-2 (82M parameters)
- **Alternative:** GPT-2 (124M parameters)
- **Fine-tuned on:** 500 synthetic world descriptions
- **Training time:** 5 epochs (~160 iterations)
- **Output:** JSON world description (500-800 tokens)

### Training Data Generation
1. Random selection from predefined components
2. Procedural combination of biomes, terrain, objects
3. Consistent JSON schema enforcement
4. Diverse color palettes and scales

### Generation Pipeline
1. User provides text prompt
2. Model extends prompt into JSON
3. JSON parsed and validated
4. World data returned to frontend
5. Three.js renders 3D scene

## API Reference

### POST /api/generate-world

**Request:**
```json
{
  "prompt": "Generate a desert world with flat terrain:"
}
```

**Response (Success):**
```json
{
  "world": {
    "name": "Desert Flat World",
    "biome": "desert",
    "terrain": "flat",
    "terrain_objects": [...],
    "structures": [...],
    "lights": [...]
  }
}
```

**Response (Error - Model Not Trained):**
```json
{
  "error": "World generator model not found",
  "message": "Please run: python world_generator_train.py",
  "path": "ai_training/world_generator/models/world_generator_final"
}
```

## Files Created

```
world_generator_train.py          # Training script
generate_world.py                 # Standalone generation script
public/world_generator.html       # Web interface
ai_training/world_generator/
  ├── data/
  │   └── world_training.jsonl    # Generated training data
  └── models/
      └── world_generator_final/  # Trained model
          ├── config.json
          ├── model.safetensors
          ├── tokenizer.json
          └── vocab.json
```

## Next Steps

1. ✅ Train the model: `python world_generator_train.py`
2. 🌐 Try web interface: http://localhost:3000/world_generator.html
3. 🎮 Integrate with game.js
4. 🎨 Customize training data for your style
5. 🚀 Deploy and share!

## Example Prompts

- "Generate a mystical forest world with glowing trees:"
- "Generate an underwater alien world with bioluminescent structures:"
- "Generate a post-apocalyptic desert world with ruins:"
- "Generate a cyberpunk city world with neon lights:"
- "Generate a medieval castle world with towers:"

Happy world generating! 🌍✨
