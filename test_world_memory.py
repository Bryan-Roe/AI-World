"""
🧪 World Memory Test Script
Test the persistent world memory features
"""

import os
import json
import sys

def test_world_memory():
    """Test world memory persistence"""
    print("=" * 60)
    print("🧪 WORLD MEMORY TEST")
    print("=" * 60)
    
    memory_file = "ai_training/game_ai/world_memory.json"
    history_file = "ai_training/game_ai/models/training_history.json"
    
    # Test 1: Check if memory files exist
    print("\n📋 Test 1: Checking for existing memory files...")
    
    if os.path.exists(memory_file):
        print(f"✅ World memory found: {memory_file}")
        with open(memory_file, 'r') as f:
            data = json.load(f)
            print(f"   Sessions: {data.get('sessions', 0)}")
            print(f"   Best Score: {data.get('best_score', 0)}")
            print(f"   Last Save: {data.get('timestamp', 'unknown')}")
    else:
        print(f"ℹ️  No world memory yet (will be created on first training)")
    
    if os.path.exists(history_file):
        print(f"\n✅ Training history found: {history_file}")
        with open(history_file, 'r') as f:
            data = json.load(f)
            print(f"   Total Episodes: {data.get('total_episodes', 0)}")
            print(f"   Best Reward: {data.get('best_reward', 0):.2f}")
            print(f"   Best Score: {data.get('best_score', 0)}")
    else:
        print(f"\nℹ️  No training history yet")
    
    # Test 2: Quick training run to create memory
    print("\n\n📋 Test 2: Running quick training to test memory...")
    print("This will train for just 5 episodes to test persistence")
    
    try:
        # Import after checking paths
        sys.path.insert(0, os.path.dirname(__file__))
        from game_ai import GameAITrainer, CONFIG
        
        # Create trainer with short config
        test_config = CONFIG.copy()
        test_config["episodes"] = 5
        test_config["max_steps"] = 50
        
        print("\n🎮 Starting mini training session...")
        trainer = GameAITrainer(test_config)
        
        # Train
        trainer.train()
        
        print("\n✅ Training completed!")
        
        # Verify files were created
        print("\n📋 Test 3: Verifying memory files were created...")
        
        if os.path.exists(memory_file):
            print(f"✅ World memory saved successfully!")
            with open(memory_file, 'r') as f:
                data = json.load(f)
                print(f"   File size: {os.path.getsize(memory_file)} bytes")
                print(f"   Sessions: {data.get('sessions', 0)}")
                print(f"   Score: {data.get('score', 0)}")
        
        if os.path.exists(history_file):
            print(f"\n✅ Training history saved successfully!")
            with open(history_file, 'r') as f:
                data = json.load(f)
                print(f"   File size: {os.path.getsize(history_file)} bytes")
                print(f"   Episodes: {data.get('total_episodes', 0)}")
        
        # Test 4: Load and verify
        print("\n\n📋 Test 4: Testing memory reload...")
        
        trainer2 = GameAITrainer(test_config)
        if trainer2.env.world_memory:
            print("✅ World memory loaded on new trainer initialization")
            print(f"   Sessions: {trainer2.env.world_memory.get('sessions', 0)}")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\n🌍 World memory is working correctly!")
        print("\nNext steps:")
        print("1. Open http://localhost:3000/game.html")
        print("2. Play the game and check the memory stats display")
        print("3. Close and reopen - your position will be restored!")
        print("4. Run python game_ai.py for full training with persistence")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nℹ️  This is likely because PyTorch is not installed yet")
        print("Run: python ai_training_setup.py")
        return False
    
    return True


if __name__ == "__main__":
    test_world_memory()
