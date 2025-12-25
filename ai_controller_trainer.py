#!/usr/bin/env python3
"""
AI Controller Trainer
Train and optimize the AI controller using reinforcement learning
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path

# Configuration
CONFIG = {
    'state_size': 32,  # Agent state representation size
    'action_size': 8,  # Number of possible actions
    'hidden_size': 128,
    'learning_rate': 0.001,
    'gamma': 0.99,  # Discount factor
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'batch_size': 64,
    'memory_size': 10000,
    'target_update': 10,
    'episodes': 1000,
    'model_dir': 'ai_training/controller/models',
    'data_dir': 'ai_training/controller/data'
}

# Action mapping
ACTIONS = {
    0: 'move_forward',
    1: 'move_backward',
    2: 'turn_left',
    3: 'turn_right',
    4: 'approach',
    5: 'flee',
    6: 'rest',
    7: 'investigate'
}


class DQN(nn.Module):
    """Deep Q-Network for agent decision making"""
    
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory:
    """Experience replay buffer"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
        
    def __len__(self):
        return len(self.memory)


class AIControllerTrainer:
    """Train the AI controller using DQN"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(config['model_dir'], exist_ok=True)
        os.makedirs(config['data_dir'], exist_ok=True)
        
        # Initialize networks
        self.policy_net = DQN(
            config['state_size'],
            config['action_size'],
            config['hidden_size']
        ).to(self.device)
        
        self.target_net = DQN(
            config['state_size'],
            config['action_size'],
            config['hidden_size']
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config['learning_rate']
        )
        
        # Replay memory
        self.memory = ReplayMemory(config['memory_size'])
        
        # Training state
        self.epsilon = config['epsilon_start']
        self.episode = 0
        self.total_steps = 0
        self.training_history = []
        
    def create_state_vector(self, agent_data):
        """Convert agent state to neural network input"""
        state = []
        
        # Position (3)
        state.extend([
            agent_data.get('position_x', 0) / 100,
            agent_data.get('position_y', 0) / 100,
            agent_data.get('position_z', 0) / 100
        ])
        
        # Health and energy (2)
        state.extend([
            agent_data.get('health', 100) / 100,
            agent_data.get('energy', 100) / 100
        ])
        
        # Velocity (3)
        velocity = agent_data.get('velocity', {'x': 0, 'y': 0, 'z': 0})
        state.extend([
            velocity.get('x', 0),
            velocity.get('y', 0),
            velocity.get('z', 0)
        ])
        
        # Perception (8)
        perception = agent_data.get('perception', {})
        state.extend([
            len(perception.get('nearbyAgents', [])) / 10,
            len(perception.get('nearbyObjects', [])) / 10,
            len(perception.get('threats', [])) / 5,
            len(perception.get('opportunities', [])) / 5,
            agent_data.get('distance_to_goal', 50) / 50,
            agent_data.get('distance_to_nearest_agent', 50) / 50,
            agent_data.get('distance_to_nearest_threat', 50) / 50,
            agent_data.get('distance_to_nearest_opportunity', 50) / 50
        ])
        
        # Mood encoding (4)
        mood = agent_data.get('mood', 'neutral')
        mood_vector = [0, 0, 0, 0]
        mood_map = {'neutral': 0, 'alert': 1, 'stressed': 2, 'curious': 3}
        if mood in mood_map:
            mood_vector[mood_map[mood]] = 1
        state.extend(mood_vector)
        
        # Goals (4)
        goals = agent_data.get('goals', [])
        state.extend([
            len(goals) / 5,
            goals[0].get('priority', 0) / 10 if goals else 0,
            goals[0].get('distance', 50) / 50 if goals else 0,
            1 if goals and goals[0].get('type') == 'survival' else 0
        ])
        
        # Recent action history (8)
        history = agent_data.get('action_history', [0] * 8)
        state.extend(history[:8])
        
        # Pad or trim to state_size
        while len(state) < self.config['state_size']:
            state.append(0)
        state = state[:self.config['state_size']]
        
        return np.array(state, dtype=np.float32)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.config['action_size'])
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def calculate_reward(self, agent_data, action, next_agent_data):
        """Calculate reward for the action taken"""
        reward = 0
        
        # Survival rewards
        if next_agent_data.get('health', 100) > agent_data.get('health', 100):
            reward += 1
        elif next_agent_data.get('health', 100) < agent_data.get('health', 100):
            reward -= 2
            
        # Energy management
        energy_diff = next_agent_data.get('energy', 100) - agent_data.get('energy', 100)
        if action == 6:  # rest
            reward += energy_diff * 0.1
        elif agent_data.get('energy', 100) < 20 and action != 6:
            reward -= 1
            
        # Goal progress
        if 'distance_to_goal' in agent_data and 'distance_to_goal' in next_agent_data:
            progress = agent_data['distance_to_goal'] - next_agent_data['distance_to_goal']
            reward += progress * 0.5
            
        # Threat avoidance
        if 'distance_to_nearest_threat' in agent_data and 'distance_to_nearest_threat' in next_agent_data:
            if next_agent_data['distance_to_nearest_threat'] > agent_data['distance_to_nearest_threat']:
                reward += 0.5
            elif next_agent_data['distance_to_nearest_threat'] < 5:
                reward -= 2
                
        # Exploration reward
        if action in [0, 1, 2, 3, 7]:  # movement actions
            reward += 0.1
            
        # Penalize excessive movement when low energy
        if agent_data.get('energy', 100) < 30 and action in [0, 1, 2, 3]:
            reward -= 0.5
            
        return reward
    
    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.config['batch_size']:
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config['batch_size']
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config['gamma'] * next_q_values
            
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self, episode_data):
        """Train on one episode of collected data"""
        episode_reward = 0
        episode_loss = []
        
        for step_data in episode_data:
            state = self.create_state_vector(step_data['state'])
            action = step_data['action']
            next_state = self.create_state_vector(step_data['next_state'])
            reward = self.calculate_reward(
                step_data['state'],
                action,
                step_data['next_state']
            )
            done = step_data.get('done', False)
            
            # Store in replay memory
            self.memory.push(state, action, reward, next_state, done)
            episode_reward += reward
            
            # Optimize
            loss = self.optimize_model()
            if loss is not None:
                episode_loss.append(loss)
            
            self.total_steps += 1
        
        # Decay epsilon
        self.epsilon = max(
            self.config['epsilon_end'],
            self.epsilon * self.config['epsilon_decay']
        )
        
        # Update target network
        if self.episode % self.config['target_update'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.episode += 1
        
        return {
            'episode': self.episode,
            'reward': episode_reward,
            'loss': np.mean(episode_loss) if episode_loss else 0,
            'epsilon': self.epsilon,
            'steps': self.total_steps
        }
    
    def train_from_file(self, data_file):
        """Train from collected trajectory data"""
        print(f"\nTraining from: {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        episodes = data.get('episodes', [])
        print(f"Found {len(episodes)} episodes")
        
        for episode_data in episodes:
            result = self.train_episode(episode_data)
            self.training_history.append(result)
            
            if result['episode'] % 10 == 0:
                print(f"Episode {result['episode']}: "
                      f"Reward={result['reward']:.2f}, "
                      f"Loss={result['loss']:.4f}, "
                      f"ε={result['epsilon']:.3f}")
    
    def train_simulation(self, num_episodes=100):
        """Train using simulated environment"""
        print(f"\nTraining in simulation for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            # Simulate episode
            episode_data = self.simulate_episode()
            result = self.train_episode(episode_data)
            self.training_history.append(result)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {result['episode']}: "
                      f"Reward={result['reward']:.2f}, "
                      f"Loss={result['loss']:.4f}, "
                      f"ε={result['epsilon']:.3f}")
                
                # Save checkpoint
                self.save_checkpoint()
    
    def simulate_episode(self, max_steps=100):
        """Simulate one episode for training"""
        episode_data = []
        
        # Initialize agent state
        state = {
            'position_x': np.random.uniform(-50, 50),
            'position_y': 0,
            'position_z': np.random.uniform(-50, 50),
            'health': 100,
            'energy': 100,
            'velocity': {'x': 0, 'y': 0, 'z': 0},
            'perception': {
                'nearbyAgents': [],
                'nearbyObjects': [],
                'threats': [],
                'opportunities': []
            },
            'mood': 'neutral',
            'goals': [{'priority': 5, 'distance': 20, 'type': 'explore'}],
            'action_history': [0] * 8
        }
        
        # Goal position
        goal_x = np.random.uniform(-50, 50)
        goal_z = np.random.uniform(-50, 50)
        
        for step in range(max_steps):
            # Calculate distances
            state['distance_to_goal'] = np.sqrt(
                (state['position_x'] - goal_x) ** 2 +
                (state['position_z'] - goal_z) ** 2
            )
            
            # Add some nearby agents randomly
            if np.random.random() < 0.3:
                state['perception']['nearbyAgents'] = [{}] * np.random.randint(1, 4)
                
            # Random threat
            if np.random.random() < 0.1:
                state['perception']['threats'] = [{}]
                state['distance_to_nearest_threat'] = np.random.uniform(5, 20)
                state['mood'] = 'alert'
            else:
                state['distance_to_nearest_threat'] = 50
            
            # Select action
            state_vector = self.create_state_vector(state)
            action = self.select_action(state_vector)
            
            # Simulate action effects
            next_state = state.copy()
            next_state['perception'] = state['perception'].copy()
            next_state['velocity'] = state['velocity'].copy()
            
            # Update based on action
            if action == 0:  # move_forward
                dx = (goal_x - state['position_x']) / state['distance_to_goal']
                dz = (goal_z - state['position_z']) / state['distance_to_goal']
                next_state['position_x'] += dx * 2
                next_state['position_z'] += dz * 2
                next_state['energy'] -= 1
            elif action == 1:  # move_backward
                dx = (goal_x - state['position_x']) / state['distance_to_goal']
                dz = (goal_z - state['position_z']) / state['distance_to_goal']
                next_state['position_x'] -= dx * 2
                next_state['position_z'] -= dz * 2
                next_state['energy'] -= 1
            elif action == 5:  # flee
                if state['distance_to_nearest_threat'] < 50:
                    next_state['position_x'] += np.random.uniform(-3, 3)
                    next_state['position_z'] += np.random.uniform(-3, 3)
                    next_state['energy'] -= 2
            elif action == 6:  # rest
                next_state['energy'] = min(100, state['energy'] + 5)
            
            # Random events
            if np.random.random() < 0.05:
                next_state['health'] -= 5
            
            # Check if done
            done = (next_state['distance_to_goal'] < 2 or 
                   next_state['health'] <= 0 or
                   step == max_steps - 1)
            
            episode_data.append({
                'state': state,
                'action': action,
                'next_state': next_state,
                'done': done
            })
            
            if done:
                break
                
            state = next_state
        
        return episode_data
    
    def save_checkpoint(self, filename=None):
        """Save model checkpoint"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"controller_model_{timestamp}.pth"
        
        filepath = os.path.join(self.config['model_dir'], filename)
        
        checkpoint = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
        
        # Save as latest
        latest_path = os.path.join(self.config['model_dir'], 'controller_latest.pth')
        torch.save(checkpoint, latest_path)
        
        return filepath
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.episode = checkpoint['episode']
        self.total_steps = checkpoint['total_steps']
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Resuming from episode {self.episode}, step {self.total_steps}")
    
    def export_for_deployment(self, output_path):
        """Export trained model for use in JavaScript"""
        # Export model weights as JSON
        state_dict = self.policy_net.state_dict()
        
        weights = {}
        for key, tensor in state_dict.items():
            weights[key] = tensor.cpu().numpy().tolist()
        
        export_data = {
            'weights': weights,
            'config': self.config,
            'actions': ACTIONS,
            'episode': self.episode,
            'total_steps': self.total_steps
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Model exported: {output_path}")


def main():
    """Main training function"""
    print("=== AI Controller Trainer ===")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    trainer = AIControllerTrainer(CONFIG)
    
    # Check for existing checkpoint
    latest_checkpoint = os.path.join(CONFIG['model_dir'], 'controller_latest.pth')
    if os.path.exists(latest_checkpoint):
        print(f"\nFound checkpoint: {latest_checkpoint}")
        response = input("Load checkpoint? (y/n): ")
        if response.lower() == 'y':
            trainer.load_checkpoint(latest_checkpoint)
    
    # Check for training data
    data_dir = Path(CONFIG['data_dir'])
    if data_dir.exists():
        data_files = list(data_dir.glob('*.json'))
        if data_files:
            print(f"\nFound {len(data_files)} data files")
            for data_file in data_files:
                trainer.train_from_file(data_file)
    
    # Train in simulation
    print("\n=== Starting Simulation Training ===")
    trainer.train_simulation(num_episodes=CONFIG['episodes'])
    
    # Save final model
    model_path = trainer.save_checkpoint('controller_final.pth')
    
    # Export for deployment
    export_path = os.path.join(CONFIG['model_dir'], 'controller_weights.json')
    trainer.export_for_deployment(export_path)
    
    print("\n=== Training Complete ===")
    print(f"Final epsilon: {trainer.epsilon:.3f}")
    print(f"Total episodes: {trainer.episode}")
    print(f"Total steps: {trainer.total_steps}")
    
    # Print summary statistics
    if trainer.training_history:
        recent = trainer.training_history[-100:]
        avg_reward = np.mean([r['reward'] for r in recent])
        avg_loss = np.mean([r['loss'] for r in recent])
        print(f"\nLast 100 episodes:")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average loss: {avg_loss:.4f}")


if __name__ == '__main__':
    main()
