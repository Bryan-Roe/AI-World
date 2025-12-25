# 🌍 World Memory - Persistent State Across Runs

## Overview

Your 3D game world and AI training now **remember everything** between runs! Player progress, world configurations, training statistics, and more are automatically saved and restored.

---

## 🎮 Browser Game World Memory

### What Gets Saved

- **Player Position & Velocity**: Resume exactly where you left off
- **World Configuration**: Current terrain, objects, and lighting
- **Score & Statistics**: Track your progress over time
- **Playtime**: Total time spent in the world
- **Session Count**: Number of times you've played

### How It Works

**Automatic Saving:**
- Auto-saves every 30 seconds while playing
- Saves on page close/refresh
- Manual save with the "💾 Save" button

**Loading:**
- Automatically loads last saved state on page load
- Restores player position and world objects
- Shows memory statistics in top-right corner

### UI Controls

```
💾 Save Button    - Manually save current world state
🗑️ Clear Button  - Reset all saved memory (with confirmation)
```

### Memory Stats Display

Real-time display in top-right corner shows:
- Total sessions played
- Total playtime (hours:minutes)
- Current score
- Player position (x, y, z)

### Browser Storage

Uses **localStorage** - data persists until you:
- Click "Clear" button
- Clear browser data
- Use different browser

---

## 🤖 AI Training World Memory

### What Gets Saved

**World State (`ai_training/game_ai/world_memory.json`):**
- Player position and velocity
- Target location
- Obstacle positions
- Collectible locations
- Current score and steps
- Best score achieved
- Total episodes trained
- Session count

**Training History (`ai_training/game_ai/models/training_history.json`):**
- Episode-by-episode rewards
- Score progression
- Epsilon decay values
- Best reward/score achieved
- Training configuration
- Timestamp of last training

### How It Works

**During Training:**
- Saves world state every 50 episodes
- Saves on training completion
- Saves on Ctrl+C interrupt
- Saves training history with statistics

**Loading:**
- Automatically loads previous world state on startup
- Shows previous training statistics
- Can restore from last checkpoint

### Python API

```python
# In your training script
env = GameEnvironment(world_size=200)

# Manual save
env.save_world_memory()

# Load and restore
env.load_world_memory()
env.restore_from_memory()  # Apply saved state

# Access memory data
print(env.world_memory)
```

### File Locations

```
ai_training/
└── game_ai/
    ├── world_memory.json          # Current world state
    └── models/
        └── training_history.json  # Training statistics
```

---

## 📊 Memory Data Structure

### Browser Memory (localStorage)

```json
{
  "playerPosition": {"x": 0, "y": 10, "z": 30},
  "playerVelocity": {"x": 0, "y": 0, "z": 0},
  "worldData": {
    "timestamp": 1703000000000,
    "objects": [...],
    "lights": [...]
  },
  "score": 0,
  "playtime": 3600,
  "sessionStart": 1703000000000,
  "sessions": 5
}
```

### AI World Memory (JSON)

```json
{
  "timestamp": "2025-12-22T10:30:00",
  "player_position": [0.0, 10.0, 30.0],
  "player_velocity": [0.0, 0.0, 0.0],
  "target_position": [25.5, 0.0, -15.2],
  "obstacles": [[...], [...]],
  "collectibles": [[...], [...]],
  "score": 25,
  "steps": 450,
  "total_episodes": 500,
  "total_rewards": 12500.5,
  "best_score": 32,
  "sessions": 3
}
```

### Training History (JSON)

```json
{
  "timestamp": "2025-12-22T10:30:00",
  "config": {...},
  "history": {
    "episode": [1, 2, 3, ...],
    "reward": [12.5, 15.2, ...],
    "score": [5, 8, ...],
    "epsilon": [1.0, 0.995, ...]
  },
  "world_memory": {...},
  "total_episodes": 500,
  "best_reward": 85.5,
  "best_score": 32
}
```

---

## 🔄 Use Cases

### 1. Continuous Training
Train your AI in multiple sessions without losing progress:
```bash
# Day 1: Train 500 episodes
python game_ai.py

# Day 2: Continue from checkpoint
python game_ai.py  # Loads previous state automatically
```

### 2. World Exploration
Explore the 3D world across sessions:
- Build complex worlds
- Save interesting configurations
- Return to favorite locations
- Track exploration time

### 3. Progress Tracking
Monitor long-term statistics:
- Total playtime across all sessions
- Best scores achieved
- Training improvement over time
- Session-by-session comparison

### 4. Experimentation
Test different configurations:
```python
# Try different algorithms
CONFIG["algorithm"] = "ppo"
# Previous training history preserved
```

---

## 🛠️ Advanced Usage

### Clear All Memory (Browser)
```javascript
// In browser console
window.game.clearWorldMemory();
```

### Clear All Memory (Python)
```bash
# Delete memory files
rm ai_training/game_ai/world_memory.json
rm ai_training/game_ai/models/training_history.json
```

### Export World State
```javascript
// Save world configuration
window.game.saveCurrentWorld();

// Access raw memory data
console.log(window.game.worldMemory);
```

### Resume Training From Checkpoint
```python
# Load specific checkpoint
trainer = GameAITrainer()
trainer.load_training_history()

# Continue training with loaded state
trainer.train()
```

---

## 📈 Benefits

✅ **Never Lose Progress**: Training and exploration state persists
✅ **Track Improvement**: See statistics across all sessions
✅ **Save Time**: Resume exactly where you left off
✅ **Experiment Safely**: Try changes without losing previous work
✅ **Long-Term Training**: Split training across multiple days
✅ **Performance Analysis**: Compare results over time

---

## 🔍 Troubleshooting

### Memory Not Saving (Browser)

**Issue**: World state doesn't persist
**Solutions**:
- Check browser console for errors
- Ensure localStorage is enabled
- Check browser storage quota
- Try incognito mode to test

### Memory Not Loading (Python)

**Issue**: Training history not found
**Solutions**:
```bash
# Check if files exist
ls ai_training/game_ai/world_memory.json
ls ai_training/game_ai/models/training_history.json

# Verify JSON is valid
python -c "import json; print(json.load(open('ai_training/game_ai/world_memory.json')))"
```

### Clear Corrupted Memory

**Browser:**
```javascript
localStorage.removeItem('worldMemory');
```

**Python:**
```bash
rm ai_training/game_ai/world_memory.json
```

---

## 🎯 Best Practices

1. **Regular Saves**: Let auto-save handle it, but manually save before major changes
2. **Backup Important Worlds**: Copy memory files for interesting configurations
3. **Monitor Stats**: Check memory display to track progress
4. **Clear When Needed**: Reset memory when starting fresh experiments
5. **Version Control**: Add memory files to `.gitignore` for personal saves

---

## 🚀 Quick Start

### Browser Game
1. Open http://localhost:3000/game.html
2. Play the game (auto-saves every 30s)
3. Close and reopen - you're back where you left off!
4. Check stats in top-right corner

### AI Training
1. Run training: `python game_ai.py`
2. Stop anytime (Ctrl+C)
3. Run again - continues from last state
4. View history in `training_history.json`

---

## 📝 Summary

**World Memory** transforms your 3D game and AI training into a **persistent experience**. Every session builds on the last, creating a continuous journey of exploration and learning.

🌍 **Your world remembers everything!**

---

*Last Updated: December 22, 2025*
