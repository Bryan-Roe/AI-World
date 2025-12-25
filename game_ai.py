"""
🎮 Game AI - Reinforcement Learning Agent
Train an AI to navigate and interact with your 3D world

USAGE:
1. Run: python game_ai.py
2. The AI will learn to navigate, collect rewards, and avoid obstacles

This integrates with your 3D game world!
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
import json
from datetime import datetime
import matplotlib.pyplot as plt

# ============== CONFIGURATION ==============
CONFIG = {
    "model_dir": "ai_training/game_ai/models",
    
    # Environment settings
    "world_size": 200,  # Match your 3D world
    "state_dim": 12,  # Position(3) + Velocity(3) + NearestObjects(6)
    "action_dim": 6,  # Forward, Backward, Left, Right, Jump, Interact
    
    # Training settings
    "episodes": 1000,
    "max_steps": 500,
    "batch_size": 64,
    "gamma": 0.99,  # Discount factor
    "lr": 3e-4,
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_decay": 0.995,
    
    # Network settings
    "hidden_size": 256,
    
    # Memory
    "memory_size": 100000,
    
    # Algorithm: "dqn", "ppo", "a2c"
    "algorithm": "ppo",
}


class GameEnvironment:
    """
    Simulated 3D game environment
    This mimics your Three.js world for training
    """
    def __init__(self, world_size=200, memory_file=None):
        self.world_size = world_size
        self.memory_file = memory_file or "ai_training/game_ai/world_memory.json"
        self.world_memory = self.load_world_memory()
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.player_pos = np.array([0.0, 10.0, 30.0])  # x, y, z
        self.player_vel = np.array([0.0, 0.0, 0.0])
        self.target_pos = self._random_position()
        self.obstacles = [self._random_position() for _ in range(20)]
        self.collectibles = [self._random_position() for _ in range(10)]
        self.steps = 0
        self.score = 0
        return self._get_state()
    
    def _random_position(self):
        """Generate random position in world"""
        return np.array([
            (random.random() - 0.5) * self.world_size,
            0,
            (random.random() - 0.5) * self.world_size
        ])
    
    def _get_state(self):
        """Get current state observation"""
        # Normalize positions
        norm_pos = self.player_pos / self.world_size
        norm_vel = np.clip(self.player_vel / 10.0, -1, 1)
        
        # Distance to nearest objects
        nearest_obstacles = self._get_nearest_distances(self.obstacles, 3)
        nearest_collectibles = self._get_nearest_distances(self.collectibles, 3)
        
        state = np.concatenate([
            norm_pos,
            norm_vel,
            nearest_obstacles,
            nearest_collectibles
        ])
        return state.astype(np.float32)
    
    def _get_nearest_distances(self, objects, n):
        """Get normalized distances to n nearest objects"""
        if not objects:
            return np.zeros(n)
        
        distances = [np.linalg.norm(self.player_pos - obj) for obj in objects]
        distances.sort()
        distances = distances[:n]
        
        # Pad if needed
        while len(distances) < n:
            distances.append(self.world_size)
        
        # Normalize
        return np.array(distances) / self.world_size
    
    def step(self, action):
        """Execute action and return new state, reward, done"""
        self.steps += 1
        reward = -0.01  # Small penalty for each step (encourages efficiency)
        
        # Action mapping
        move_speed = 2.0
        actions = {
            0: np.array([0, 0, -move_speed]),   # Forward
            1: np.array([0, 0, move_speed]),    # Backward
            2: np.array([-move_speed, 0, 0]),   # Left
            3: np.array([move_speed, 0, 0]),    # Right
            4: np.array([0, 5.0, 0]),           # Jump
            5: np.array([0, 0, 0]),             # Interact/Stay
        }
        
        # Apply action
        self.player_vel += actions.get(action, np.zeros(3))
        self.player_vel *= 0.9  # Friction
        
        # Update position
        self.player_pos += self.player_vel
        
        # Gravity
        if self.player_pos[1] > 0:
            self.player_vel[1] -= 0.5
        else:
            self.player_pos[1] = 0
            self.player_vel[1] = 0
        
        # World bounds
        self.player_pos[0] = np.clip(self.player_pos[0], -self.world_size/2, self.world_size/2)
        self.player_pos[2] = np.clip(self.player_pos[2], -self.world_size/2, self.world_size/2)
        
        # Check collectibles
        for i, collectible in enumerate(self.collectibles[:]):
            if np.linalg.norm(self.player_pos - collectible) < 5:
                reward += 10.0
                self.score += 1
                self.collectibles[i] = self._random_position()
        
        # Check obstacles
        for obstacle in self.obstacles:
            if np.linalg.norm(self.player_pos - obstacle) < 3:
                reward -= 5.0
        
        # Target bonus
        dist_to_target = np.linalg.norm(self.player_pos - self.target_pos)
        if dist_to_target < 10:
            reward += 50.0
            self.target_pos = self._random_position()
        
        done = self.steps >= CONFIG["max_steps"]
        
        return self._get_state(), reward, done, {"score": self.score}
    
    def render(self):
        """Return state for visualization"""
        return {
            "player": self.player_pos.tolist(),
            "target": self.target_pos.tolist(),
            "obstacles": [o.tolist() for o in self.obstacles],
            "collectibles": [c.tolist() for c in self.collectibles],
            "score": self.score
        }
    
    def save_world_memory(self):
        """Save world state to persistent memory"""
        try:
            memory = {
                "timestamp": datetime.now().isoformat(),
                "player_position": self.player_pos.tolist(),
                "player_velocity": self.player_vel.tolist(),
                "target_position": self.target_pos.tolist(),
                "obstacles": [o.tolist() for o in self.obstacles],
                "collectibles": [c.tolist() for c in self.collectibles],
                "score": self.score,
                "steps": self.steps,
                "total_episodes": self.world_memory.get("total_episodes", 0),
                "total_rewards": self.world_memory.get("total_rewards", 0),
                "best_score": max(self.score, self.world_memory.get("best_score", 0)),
                "sessions": self.world_memory.get("sessions", 0) + 1
            }
            
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(memory, f, indent=2)
            
            return memory
        except Exception as e:
            print(f"⚠️ Failed to save world memory: {e}")
            return {}
    
    def load_world_memory(self):
        """Load world state from persistent memory"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory = json.load(f)
                    print(f"📂 World memory loaded:")
                    print(f"   Sessions: {memory.get('sessions', 0)}")
                    print(f"   Best Score: {memory.get('best_score', 0)}")
                    print(f"   Total Episodes: {memory.get('total_episodes', 0)}")
                    return memory
        except Exception as e:
            print(f"⚠️ Failed to load world memory: {e}")
        
        return {"sessions": 0, "best_score": 0, "total_episodes": 0, "total_rewards": 0}
    
    def restore_from_memory(self):
        """Restore world state from memory"""
        if self.world_memory:
            try:
                if "player_position" in self.world_memory:
                    self.player_pos = np.array(self.world_memory["player_position"])
                if "player_velocity" in self.world_memory:
                    self.player_vel = np.array(self.world_memory["player_velocity"])
                if "target_position" in self.world_memory:
                    self.target_pos = np.array(self.world_memory["target_position"])
                if "obstacles" in self.world_memory:
                    self.obstacles = [np.array(o) for o in self.world_memory["obstacles"]]
                if "collectibles" in self.world_memory:
                    self.collectibles = [np.array(c) for c in self.world_memory["collectibles"]]
                if "score" in self.world_memory:
                    self.score = self.world_memory["score"]
                
                print(f"✅ World state restored from memory")
                return True
            except Exception as e:
                print(f"⚠️ Failed to restore world state: {e}")
        return False


class DQNetwork(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    """Actor-Critic Network for PPO/A2C"""
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)
    
    def get_action(self, state):
        probs, value = self(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class GameAITrainer:
    def __init__(self, config=CONFIG):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        os.makedirs(config["model_dir"], exist_ok=True)
        
        self.env = GameEnvironment(config["world_size"])
        self.training_history_file = os.path.join(config["model_dir"], "training_history.json")
        self.load_training_history()
        self.history = {"episode": [], "reward": [], "score": [], "epsilon": []}
        
        # Initialize based on algorithm
        if config["algorithm"] == "dqn":
            self._init_dqn()
        else:
            self._init_ppo()
    
    def _init_dqn(self):
        """Initialize DQN components"""
        self.policy_net = DQNetwork(
            self.config["state_dim"],
            self.config["action_dim"],
            self.config["hidden_size"]
        ).to(self.device)
        
        self.target_net = DQNetwork(
            self.config["state_dim"],
            self.config["action_dim"],
            self.config["hidden_size"]
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config["lr"])
        self.memory = ReplayBuffer(self.config["memory_size"])
        self.epsilon = self.config["eps_start"]
    
    def _init_ppo(self):
        """Initialize PPO components"""
        self.model = ActorCritic(
            self.config["state_dim"],
            self.config["action_dim"],
            self.config["hidden_size"]
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.epsilon = 0  # Not used in PPO
    
    def select_action_dqn(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.config["action_dim"])
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()
    
    def train_dqn(self):
        """Train using DQN algorithm"""
        print(f"\n🚀 Starting DQN Training for {self.config['episodes']} episodes...\n")
        
        for episode in range(self.config["episodes"]):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(self.config["max_steps"]):
                action = self.select_action_dqn(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # Train on batch
                if len(self.memory) >= self.config["batch_size"]:
                    self._update_dqn()
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(
                self.config["eps_end"],
                self.epsilon * self.config["eps_decay"]
            )
            
            # Update target network
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Log progress
            self.history["episode"].append(episode)
            self.history["reward"].append(total_reward)
            self.history["score"].append(info["score"])
            self.history["epsilon"].append(self.epsilon)
            
            # Save world memory every 50 episodes
            if episode % 50 == 0:
                self.env.save_world_memory()
                self.save_training_history()
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.history["reward"][-10:])
                print(f"Episode {episode:4d} | Reward: {total_reward:7.2f} | "
                      f"Avg: {avg_reward:7.2f} | Score: {info['score']:3d} | "
                      f"ε: {self.epsilon:.3f}")
        
        self.save_model("dqn_final.pth")
        
        # Save final world state and training history
        self.env.save_world_memory()
        self.save_training_history()
        print(f"✅ World memory and training history saved")
        
        self.plot_training()
    
    def _update_dqn(self):
        """Update DQN network"""
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config["batch_size"]
        )
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config["gamma"] * next_q
        
        # Loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def train_ppo(self):
        """Train using PPO algorithm"""
        print(f"\n🚀 Starting PPO Training for {self.config['episodes']} episodes...\n")
        
        for episode in range(self.config["episodes"]):
            state = self.env.reset()
            total_reward = 0
            
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            
            for step in range(self.config["max_steps"]):
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, log_prob, value = self.model.get_action(state_t)
                
                next_state, reward, done, info = self.env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Update policy
            self._update_ppo(states, actions, rewards, log_probs, values, dones)
            
            # Log progress
            self.history["episode"].append(episode)
            self.history["reward"].append(total_reward)
            self.history["score"].append(info["score"])
            self.history["epsilon"].append(0)
            
            # Save world memory every 50 episodes
            if episode % 50 == 0:
                self.env.save_world_memory()
                self.save_training_history()
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.history["reward"][-10:])
                print(f"Episode {episode:4d} | Reward: {total_reward:7.2f} | "
                      f"Avg: {avg_reward:7.2f} | Score: {info['score']:3d}")
        
        self.save_model("ppo_final.pth")
        
        # Save final world state and training history
        self.env.save_world_memory()
        self.save_training_history()
        print(f"✅ World memory and training history saved")
        
        self.plot_training()
    
    def _update_ppo(self, states, actions, rewards, log_probs, values, dones):
        """Update PPO network"""
        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.config["gamma"] * R * (1 - dones[i])
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.cat(values).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.cat(log_probs).detach()
        
        for _ in range(4):  # PPO epochs
            probs, new_values = self.model(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            # Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped objective
            clip_eps = 0.2
            obj1 = ratio * advantages
            obj2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            
            # Loss
            policy_loss = -torch.min(obj1, obj2).mean()
            value_loss = F.mse_loss(new_values.squeeze(), returns)
            entropy_loss = -dist.entropy().mean()
            
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
    
    def train(self):
        """Main training entry point"""
        if self.config["algorithm"] == "dqn":
            self.train_dqn()
        else:
            self.train_ppo()
    
    def save_model(self, filename):
        """Save trained model"""
        path = os.path.join(self.config["model_dir"], filename)
        
        if self.config["algorithm"] == "dqn":
            torch.save({
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "config": self.config,
                "history": self.history
            }, path)
        else:
            torch.save({
                "model": self.model.state_dict(),
                "config": self.config,
                "history": self.history
            }, path)
        
        print(f"💾 Model saved to {path}")
    
    def load_model(self, filename):
        """Load trained model"""
        path = os.path.join(self.config["model_dir"], filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        if "policy_net" in checkpoint:
            self._init_dqn()
            self.policy_net.load_state_dict(checkpoint["policy_net"])
        else:
            self._init_ppo()
            self.model.load_state_dict(checkpoint["model"])
        
        self.config = checkpoint["config"]
        self.history = checkpoint["history"]
        print(f"📂 Model loaded from {path}")
    
    def play(self, episodes=5, render=True):
        """Watch the trained agent play"""
        if self.config["algorithm"] == "dqn":
            self.policy_net.eval()
        else:
            self.model.eval()
        
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(self.config["max_steps"]):
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    if self.config["algorithm"] == "dqn":
                        action = self.policy_net(state_t).argmax().item()
                    else:
                        probs, _ = self.model(state_t)
                        action = probs.argmax().item()
                
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                
                if render and step % 50 == 0:
                    env_state = self.env.render()
                    print(f"Step {step}: Pos={env_state['player']}, Score={env_state['score']}")
                
                if done:
                    break
            
            print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Score = {info['score']}")
    
    def export_for_web(self, filename="game_ai_model.json"):
        """Export model weights for use in browser (TensorFlow.js format)"""
        path = os.path.join(self.config["model_dir"], filename)
        
        if self.config["algorithm"] == "dqn":
            model = self.policy_net
        else:
            model = self.model
        
        # Export weights as JSON
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.detach().cpu().numpy().tolist()
        
        export_data = {
            "config": self.config,
            "weights": weights,
            "architecture": str(model)
        }
        
        with open(path, 'w') as f:
            json.dump(export_data, f)
        
        print(f"🌐 Model exported for web: {path}")
    
    def save_training_history(self):
        """Save training history to file"""
        try:
            history_data = {
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "history": self.history,
                "world_memory": self.env.world_memory,
                "total_episodes": len(self.history["episode"]),
                "best_reward": max(self.history["reward"]) if self.history["reward"] else 0,
                "best_score": max(self.history["score"]) if self.history["score"] else 0
            }
            
            os.makedirs(os.path.dirname(self.training_history_file), exist_ok=True)
            with open(self.training_history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            print(f"💾 Training history saved to {self.training_history_file}")
        except Exception as e:
            print(f"⚠️ Failed to save training history: {e}")
    
    def load_training_history(self):
        """Load previous training history"""
        try:
            if os.path.exists(self.training_history_file):
                with open(self.training_history_file, 'r') as f:
                    data = json.load(f)
                    print(f"📂 Training history loaded:")
                    print(f"   Previous episodes: {data.get('total_episodes', 0)}")
                    print(f"   Best reward: {data.get('best_reward', 0):.2f}")
                    print(f"   Best score: {data.get('best_score', 0)}")
                    print(f"   Last trained: {data.get('timestamp', 'unknown')}")
                    
                    # Optionally restore history to continue training
                    # self.history = data.get('history', self.history)
                    return data
        except Exception as e:
            print(f"⚠️ Failed to load training history: {e}")
        return None
    
    def plot_training(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(self.history["episode"], self.history["reward"])
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        
        # Moving average
        window = 50
        if len(self.history["reward"]) >= window:
            ma = np.convolve(self.history["reward"], np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(self.history["reward"])), ma)
        axes[0, 1].set_title(f"Reward (Moving Avg {window})")
        axes[0, 1].set_xlabel("Episode")
        
        axes[1, 0].plot(self.history["episode"], self.history["score"])
        axes[1, 0].set_title("Score per Episode")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Score")
        
        if self.config["algorithm"] == "dqn":
            axes[1, 1].plot(self.history["episode"], self.history["epsilon"])
            axes[1, 1].set_title("Epsilon Decay")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Epsilon")
        
        plt.tight_layout()
        plot_path = os.path.join(self.config["model_dir"], "training_progress.png")
        plt.savefig(plot_path)
        print(f"📈 Training plot saved to {plot_path}")
        plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("🎮 GAME AI - Reinforcement Learning")
    print("=" * 60)
    
    trainer = GameAITrainer()
    
    # Train the agent
    trainer.train()
    
    # Watch it play
    print("\n🎬 Watching trained agent play...")
    trainer.play(episodes=3)
    
    # Export for web
    trainer.export_for_web()
