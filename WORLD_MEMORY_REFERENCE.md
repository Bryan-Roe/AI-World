# 🌍 World Memory - Quick Reference

## Browser (JavaScript)

### Access Memory
```javascript
// Get current memory state
console.log(window.game.worldMemory);

// Check specific values
console.log(window.game.worldMemory.sessions);
console.log(window.game.worldMemory.playtime);
console.log(window.game.worldMemory.score);
```

### Manual Operations
```javascript
// Save world state
window.game.saveWorldMemory();

// Save world configuration
window.game.saveCurrentWorld();

// Clear all memory
window.game.clearWorldMemory();

// Start auto-save (30 seconds)
window.game.startAutoSave(30);

// Stop auto-save
window.game.stopAutoSave();

// Update stats display
window.game.displayMemoryStats();
```

### Access Raw localStorage
```javascript
// Get saved data
const memory = JSON.parse(localStorage.getItem('worldMemory'));

// Clear memory
localStorage.removeItem('worldMemory');
```

---

## Python

### Load Memory
```python
from game_ai import GameEnvironment

# Create environment (auto-loads memory)
env = GameEnvironment(world_size=200)

# Access memory data
print(env.world_memory)
print(f"Sessions: {env.world_memory.get('sessions', 0)}")
print(f"Best Score: {env.world_memory.get('best_score', 0)}")
```

### Save Memory
```python
# Save current state
env.save_world_memory()

# Restore from memory
env.restore_from_memory()
```

### Training History
```python
from game_ai import GameAITrainer

# Create trainer (auto-loads history)
trainer = GameAITrainer()

# Access history
history = trainer.load_training_history()
if history:
    print(f"Previous episodes: {history['total_episodes']}")
    print(f"Best reward: {history['best_reward']}")

# Save history
trainer.save_training_history()
```

### Access Files Directly
```python
import json

# Load world memory
with open('ai_training/game_ai/world_memory.json', 'r') as f:
    memory = json.load(f)
    print(memory)

# Load training history
with open('ai_training/game_ai/models/training_history.json', 'r') as f:
    history = json.load(f)
    print(history)
```

---

## Memory Structure

### Browser localStorage
```javascript
{
  playerPosition: {x: float, y: float, z: float},
  playerVelocity: {x: float, y: float, z: float},
  worldData: {
    timestamp: int,
    objects: [...],
    lights: [...]
  },
  score: int,
  playtime: int,        // seconds
  sessionStart: int,    // timestamp
  sessions: int
}
```

### Python world_memory.json
```python
{
  "timestamp": str,     # ISO format
  "player_position": [x, y, z],
  "player_velocity": [x, y, z],
  "target_position": [x, y, z],
  "obstacles": [[x,y,z], ...],
  "collectibles": [[x,y,z], ...],
  "score": int,
  "steps": int,
  "total_episodes": int,
  "total_rewards": float,
  "best_score": int,
  "sessions": int
}
```

### Python training_history.json
```python
{
  "timestamp": str,
  "config": {...},
  "history": {
    "episode": [1, 2, 3, ...],
    "reward": [12.5, 15.2, ...],
    "score": [5, 8, ...],
    "epsilon": [1.0, 0.995, ...]
  },
  "world_memory": {...},
  "total_episodes": int,
  "best_reward": float,
  "best_score": int
}
```

---

## Common Operations

### Check if Memory Exists
```javascript
// Browser
if (localStorage.getItem('worldMemory')) {
  console.log('Memory exists');
}
```

```python
# Python
import os
if os.path.exists('ai_training/game_ai/world_memory.json'):
    print('Memory exists')
```

### Get Session Count
```javascript
// Browser
const sessions = window.game.worldMemory.sessions;
```

```python
# Python
sessions = env.world_memory.get('sessions', 0)
```

### Get Best Score
```javascript
// Browser - stored in worldData if saved
const score = window.game.worldMemory.score;
```

```python
# Python
best_score = env.world_memory.get('best_score', 0)
```

### Reset Everything
```javascript
// Browser
window.game.clearWorldMemory();
```

```bash
# Python
rm ai_training/game_ai/world_memory.json
rm ai_training/game_ai/models/training_history.json
```

---

## Configuration

### Change Auto-Save Interval (Browser)
```javascript
// Default: 30 seconds
window.game.startAutoSave(60);  // 60 seconds
```

### Change Save Frequency (Python)
```python
# In game_ai.py, find:
if episode % 50 == 0:  # Change this number
    self.env.save_world_memory()
```

### Custom Memory Location (Python)
```python
env = GameEnvironment(
    world_size=200,
    memory_file="custom/path/memory.json"
)
```

---

## Debugging

### Check Browser Memory Size
```javascript
const memory = localStorage.getItem('worldMemory');
console.log(`Memory size: ${memory.length} characters`);
console.log(`Estimated: ${Math.round(memory.length / 1024)} KB`);
```

### Validate Python JSON
```bash
# Check if valid JSON
python -c "import json; print(json.load(open('ai_training/game_ai/world_memory.json')))"
```

### View Memory in Console
```javascript
// Browser - formatted
console.table(window.game.worldMemory);
```

```python
# Python - formatted
import json
with open('ai_training/game_ai/world_memory.json') as f:
    print(json.dumps(json.load(f), indent=2))
```

---

## Events

### Browser Events
```javascript
// Before page unload
window.addEventListener('beforeunload', () => {
  window.game.saveWorldMemory();
});

// On page load
window.addEventListener('load', () => {
  window.game.loadWorldMemory();
});
```

### Python Training Events
```python
# Training start
def __init__():
    self.load_training_history()  # Auto-loads

# Every 50 episodes
if episode % 50 == 0:
    self.env.save_world_memory()

# On completion
self.env.save_world_memory()
self.save_training_history()

# On interrupt (Ctrl+C)
except KeyboardInterrupt:
    self.env.save_world_memory()
    self.save_training_history()
```

---

## Testing

### Browser Console Test
```javascript
// 1. Check current memory
console.log(window.game.worldMemory);

// 2. Move around
// (use WASD keys)

// 3. Save manually
window.game.saveWorldMemory();

// 4. Check updated
console.log(window.game.worldMemory.playerPosition);

// 5. Refresh page
location.reload();

// 6. Check if restored
console.log(window.game.worldMemory.playerPosition);
```

### Python Test
```python
# test_memory.py
from game_ai import GameEnvironment

# 1. Create environment
env = GameEnvironment()
print(f"Initial sessions: {env.world_memory.get('sessions', 0)}")

# 2. Modify state
env.score = 100

# 3. Save
env.save_world_memory()

# 4. Create new instance
env2 = GameEnvironment()
print(f"Loaded score: {env2.world_memory.get('score', 0)}")
```

---

## File Locations

```
Browser:
  localStorage['worldMemory']

Python:
  ai_training/
    game_ai/
      world_memory.json           ← World state
      models/
        training_history.json     ← Training stats
        final_model_*.pt          ← Model checkpoints
```

---

## Quick Commands

```bash
# View memory
cat ai_training/game_ai/world_memory.json | python -m json.tool

# Check memory size
du -h ai_training/game_ai/world_memory.json

# Backup memory
cp ai_training/game_ai/world_memory.json backup.json

# Clear memory
rm ai_training/game_ai/world_memory.json

# Test implementation
python test_world_memory.py
```

---

**💡 Pro Tip**: Check the memory stats display in the browser (top-right) for real-time monitoring!
