"""
🧠 Custom Neural Network - Flexible Architecture Builder
Build and train custom neural networks for any task

USAGE:
1. Define your architecture in the CONFIG
2. Provide your data
3. Run: python custom_nn.py

Supports: Classification, Regression, Autoencoders, GANs
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import json

# ============== CONFIGURATION ==============
CONFIG = {
    "model_dir": "ai_training/custom_nn/models",
    
    # Task type: "classification", "regression", "autoencoder", "gan"
    "task": "classification",
    
    # Architecture
    "input_dim": 784,  # e.g., 28x28 image flattened
    "output_dim": 10,  # e.g., 10 classes
    "hidden_layers": [512, 256, 128],  # Hidden layer sizes
    "activation": "relu",  # relu, leaky_relu, elu, gelu, silu, tanh, sigmoid
    "dropout": 0.3,
    "batch_norm": True,
    
    # Training
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "scheduler": "cosine",  # step, cosine, plateau, none
    
    # Data
    "train_split": 0.8,
    "val_split": 0.1,
}


def get_activation(name):
    """Get activation function by name"""
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.2),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(dim=-1),
    }
    return activations.get(name, nn.ReLU())


class FlexibleNN(nn.Module):
    """
    Flexible neural network that can be configured for various tasks
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        layers = []
        input_dim = config["input_dim"]
        
        # Build hidden layers
        for hidden_dim in config["hidden_layers"]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if config["batch_norm"]:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(get_activation(config["activation"]))
            
            if config["dropout"] > 0:
                layers.append(nn.Dropout(config["dropout"]))
            
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, config["output_dim"]))
        
        # Final activation based on task
        if config["task"] == "classification" and config["output_dim"] > 1:
            pass  # Use CrossEntropyLoss which includes softmax
        elif config["task"] == "regression":
            pass  # No activation for regression
        elif config["task"] == "binary":
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class Autoencoder(nn.Module):
    """Autoencoder for unsupervised learning / dimensionality reduction"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder
        encoder_layers = []
        input_dim = config["input_dim"]
        
        for hidden_dim in config["hidden_layers"]:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(get_activation(config["activation"]))
            if config["dropout"] > 0:
                encoder_layers.append(nn.Dropout(config["dropout"]))
            input_dim = hidden_dim
        
        # Latent space
        encoder_layers.append(nn.Linear(input_dim, config["output_dim"]))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (reverse of encoder)
        decoder_layers = []
        input_dim = config["output_dim"]
        
        for hidden_dim in reversed(config["hidden_layers"]):
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            decoder_layers.append(get_activation(config["activation"]))
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(input_dim, config["input_dim"]))
        decoder_layers.append(nn.Sigmoid())  # For normalized input
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


class Generator(nn.Module):
    """Generator for GAN"""
    def __init__(self, latent_dim, output_dim, hidden_layers):
        super().__init__()
        
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.network(z)


class Discriminator(nn.Module):
    """Discriminator for GAN"""
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        
        layers = []
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CustomNNTrainer:
    def __init__(self, config=CONFIG):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        os.makedirs(config["model_dir"], exist_ok=True)
        
        self.model = None
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    def build_model(self):
        """Build model based on task"""
        task = self.config["task"]
        
        if task == "autoencoder":
            self.model = Autoencoder(self.config).to(self.device)
        elif task == "gan":
            self.generator = Generator(
                self.config["output_dim"],  # latent_dim
                self.config["input_dim"],   # output_dim
                self.config["hidden_layers"]
            ).to(self.device)
            self.discriminator = Discriminator(
                self.config["input_dim"],
                list(reversed(self.config["hidden_layers"]))
            ).to(self.device)
            self.model = self.generator  # For saving/loading
        else:
            self.model = FlexibleNN(self.config).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n🏗️  Model built: {self.config['task']}")
        print(f"   Architecture: {self.config['input_dim']} → {self.config['hidden_layers']} → {self.config['output_dim']}")
        print(f"   Total parameters: {total_params:,}")
        
        return self.model
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        print("\n📊 Creating sample data...")
        
        if self.config["task"] == "classification":
            # Create synthetic classification data
            n_samples = 5000
            n_features = self.config["input_dim"]
            n_classes = self.config["output_dim"]
            
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            # Create class-dependent patterns
            for i in range(n_classes):
                mask = np.random.rand(n_samples) < (1.0 / n_classes)
                X[mask, i * (n_features // n_classes):(i+1) * (n_features // n_classes)] += 2
            y = np.argmax(X.reshape(n_samples, n_classes, -1).mean(axis=2), axis=1)
            
            print(f"   Created {n_samples} samples, {n_features} features, {n_classes} classes")
            
        elif self.config["task"] == "regression":
            n_samples = 5000
            n_features = self.config["input_dim"]
            
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            # Create regression target
            weights = np.random.randn(n_features, self.config["output_dim"]).astype(np.float32)
            y = X @ weights + np.random.randn(n_samples, self.config["output_dim"]).astype(np.float32) * 0.1
            
            print(f"   Created {n_samples} samples for regression")
            
        elif self.config["task"] == "autoencoder":
            n_samples = 5000
            n_features = self.config["input_dim"]
            
            # Create data with patterns
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            X = (X - X.min()) / (X.max() - X.min())  # Normalize to [0, 1]
            y = X  # Autoencoder reconstructs input
            
            print(f"   Created {n_samples} samples for autoencoder")
            
        else:  # GAN
            n_samples = 5000
            n_features = self.config["input_dim"]
            
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            X = (X - X.mean()) / X.std()  # Standardize
            y = np.zeros(n_samples)  # Not used for GAN
            
            print(f"   Created {n_samples} samples for GAN")
        
        return torch.FloatTensor(X), torch.LongTensor(y) if self.config["task"] == "classification" else torch.FloatTensor(y)
    
    def prepare_data(self, X=None, y=None):
        """Prepare data loaders"""
        if X is None:
            X, y = self.create_sample_data()
        
        dataset = TensorDataset(X, y)
        n_samples = len(dataset)
        
        train_size = int(self.config["train_split"] * n_samples)
        val_size = int(self.config["val_split"] * n_samples)
        test_size = n_samples - train_size - val_size
        
        train_set, val_set, test_set = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.config["batch_size"])
        test_loader = DataLoader(test_set, batch_size=self.config["batch_size"])
        
        print(f"\n📊 Data split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        return train_loader, val_loader, test_loader
    
    def train(self, X=None, y=None):
        """Main training function"""
        self.build_model()
        train_loader, val_loader, test_loader = self.prepare_data(X, y)
        
        if self.config["task"] == "gan":
            self._train_gan(train_loader)
        else:
            self._train_standard(train_loader, val_loader)
        
        # Final evaluation
        if test_loader and self.config["task"] != "gan":
            test_loss, test_acc = self.evaluate(test_loader)
            print(f"\n📊 Test Results: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        
        self.save_model("final_model.pth")
        self.plot_history()
    
    def _train_standard(self, train_loader, val_loader):
        """Standard training loop"""
        # Loss function
        if self.config["task"] == "classification":
            criterion = nn.CrossEntropyLoss()
        elif self.config["task"] == "autoencoder":
            criterion = nn.MSELoss()
        else:
            criterion = nn.MSELoss()
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        # Scheduler
        if self.config["scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config["epochs"]
            )
        elif self.config["scheduler"] == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        elif self.config["scheduler"] == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        else:
            scheduler = None
        
        print(f"\n🚀 Starting training for {self.config['epochs']} epochs...\n")
        best_val_loss = float('inf')
        
        for epoch in range(self.config["epochs"]):
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                if self.config["task"] == "autoencoder":
                    outputs, _ = self.model(inputs)
                    loss = criterion(outputs, inputs)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if self.config["task"] == "classification":
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})
                else:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            train_loss /= len(train_loader)
            train_acc = 100. * correct / total if total > 0 else 0
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            print(f"   Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", end="")
            if self.config["task"] == "classification":
                print(f", Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
            else:
                print()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pth")
            
            # Update scheduler
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        
        print(f"\n✅ Training complete! Best validation loss: {best_val_loss:.4f}")
    
    def _train_gan(self, train_loader):
        """GAN training loop"""
        criterion = nn.BCELoss()
        
        optimizer_g = optim.Adam(self.generator.parameters(), lr=self.config["learning_rate"], betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.config["learning_rate"], betas=(0.5, 0.999))
        
        latent_dim = self.config["output_dim"]
        
        print(f"\n🚀 Starting GAN training for {self.config['epochs']} epochs...\n")
        
        for epoch in range(self.config["epochs"]):
            g_loss_total = 0.0
            d_loss_total = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for real_data, _ in pbar:
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                optimizer_d.zero_grad()
                
                # Real data
                real_output = self.discriminator(real_data)
                d_loss_real = criterion(real_output, real_labels)
                
                # Fake data
                z = torch.randn(batch_size, latent_dim).to(self.device)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()
                
                # Train Generator
                optimizer_g.zero_grad()
                
                z = torch.randn(batch_size, latent_dim).to(self.device)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data)
                g_loss = criterion(fake_output, real_labels)
                
                g_loss.backward()
                optimizer_g.step()
                
                g_loss_total += g_loss.item()
                d_loss_total += d_loss.item()
                
                pbar.set_postfix({"G_loss": f"{g_loss.item():.4f}", "D_loss": f"{d_loss.item():.4f}"})
            
            g_loss_avg = g_loss_total / len(train_loader)
            d_loss_avg = d_loss_total / len(train_loader)
            
            self.history["train_loss"].append(g_loss_avg)
            self.history["val_loss"].append(d_loss_avg)
            
            print(f"   Generator Loss: {g_loss_avg:.4f}, Discriminator Loss: {d_loss_avg:.4f}")
        
        print("\n✅ GAN training complete!")
    
    def evaluate(self, data_loader):
        """Evaluate model on data loader"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        if self.config["task"] == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if self.config["task"] == "autoencoder":
                    outputs, _ = self.model(inputs)
                    loss = criterion(outputs, inputs)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                if self.config["task"] == "classification":
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            if self.config["task"] == "autoencoder":
                outputs, latent = self.model(X)
                return outputs.cpu().numpy(), latent.cpu().numpy()
            else:
                outputs = self.model(X)
                if self.config["task"] == "classification":
                    return torch.softmax(outputs, dim=-1).cpu().numpy()
                return outputs.cpu().numpy()
    
    def generate(self, n_samples=10):
        """Generate samples (for GAN)"""
        if self.config["task"] != "gan":
            print("⚠️  Generate is only for GAN models")
            return None
        
        self.generator.eval()
        z = torch.randn(n_samples, self.config["output_dim"]).to(self.device)
        
        with torch.no_grad():
            samples = self.generator(z)
        
        return samples.cpu().numpy()
    
    def save_model(self, filename):
        """Save model"""
        path = os.path.join(self.config["model_dir"], filename)
        
        save_dict = {
            "config": self.config,
            "history": self.history,
        }
        
        if self.config["task"] == "gan":
            save_dict["generator"] = self.generator.state_dict()
            save_dict["discriminator"] = self.discriminator.state_dict()
        else:
            save_dict["model"] = self.model.state_dict()
        
        torch.save(save_dict, path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, filename):
        """Load model"""
        path = os.path.join(self.config["model_dir"], filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint["config"]
        self.history = checkpoint["history"]
        self.build_model()
        
        if self.config["task"] == "gan":
            self.generator.load_state_dict(checkpoint["generator"])
            self.discriminator.load_state_dict(checkpoint["discriminator"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        
        print(f"📂 Model loaded from {path}")
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history["train_loss"], label="Train")
        axes[0].plot(self.history["val_loss"], label="Val/Disc")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        
        if self.history["train_acc"] and self.config["task"] == "classification":
            axes[1].plot(self.history["train_acc"], label="Train")
            axes[1].plot(self.history["val_acc"], label="Val")
            axes[1].set_title("Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.config["model_dir"], "training_history.png")
        plt.savefig(plot_path)
        print(f"📈 Training history saved to {plot_path}")
        plt.show()
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"Task: {self.config['task']}")
        print(f"Input: {self.config['input_dim']}")
        print(f"Output: {self.config['output_dim']}")
        print(f"Hidden layers: {self.config['hidden_layers']}")
        print(f"Activation: {self.config['activation']}")
        print(f"Dropout: {self.config['dropout']}")
        print(f"Batch norm: {self.config['batch_norm']}")
        print("=" * 60)
        print(self.model)
        print("=" * 60)


# ============== PRESET CONFIGURATIONS ==============
PRESETS = {
    "mnist_classifier": {
        "task": "classification",
        "input_dim": 784,
        "output_dim": 10,
        "hidden_layers": [512, 256, 128],
        "activation": "relu",
        "dropout": 0.3,
        "batch_norm": True,
    },
    "image_autoencoder": {
        "task": "autoencoder",
        "input_dim": 784,
        "output_dim": 32,  # Latent dimension
        "hidden_layers": [512, 256, 128],
        "activation": "relu",
        "dropout": 0.1,
        "batch_norm": True,
    },
    "simple_gan": {
        "task": "gan",
        "input_dim": 784,
        "output_dim": 100,  # Latent dimension
        "hidden_layers": [256, 512, 1024],
        "activation": "relu",
        "dropout": 0.0,
        "batch_norm": True,
    },
    "regression": {
        "task": "regression",
        "input_dim": 10,
        "output_dim": 1,
        "hidden_layers": [64, 32],
        "activation": "relu",
        "dropout": 0.1,
        "batch_norm": False,
    },
}


if __name__ == "__main__":
    print("=" * 60)
    print("🧠 CUSTOM NEURAL NETWORK TRAINER")
    print("=" * 60)
    
    # Choose a preset or use default CONFIG
    # config = PRESETS["mnist_classifier"]
    
    trainer = CustomNNTrainer()
    trainer.summary()
    
    # Train with sample data
    trainer.train()
    
    # Example: Use with real data
    # X = np.load("your_features.npy")
    # y = np.load("your_labels.npy")
    # trainer.train(torch.FloatTensor(X), torch.LongTensor(y))
