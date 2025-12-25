"""
🖼️ Image Classifier - CNN with PyTorch
Train on your own images with GPU acceleration

USAGE:
1. Create folders in ai_training/image_classifier/data/train/ for each class
   Example: data/train/cats/, data/train/dogs/
2. Add images to each folder
3. Run: python ai_training/image_classifier.py

The model will automatically detect your classes and train!
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# ============== CONFIGURATION ==============
CONFIG = {
    "data_dir": "ai_training/image_classifier/data",
    "model_dir": "ai_training/image_classifier/models",
    "image_size": 224,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,
    "model_type": "resnet18",  # Options: resnet18, resnet34, resnet50, efficientnet, custom
    "pretrained": True,  # Use transfer learning
    "freeze_backbone": True,  # Freeze pretrained layers initially
    "unfreeze_epoch": 5,  # Unfreeze backbone after this epoch
}


class CustomCNN(nn.Module):
    """Custom CNN architecture - modify as needed"""
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImageClassifier:
    def __init__(self, config=CONFIG):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model = None
        self.classes = None
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
    def setup_data(self):
        """Setup data loaders with augmentation"""
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.config["image_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(self.config["image_size"] + 32),
            transforms.CenterCrop(self.config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dir = os.path.join(self.config["data_dir"], "train")
        val_dir = os.path.join(self.config["data_dir"], "val")
        
        # Check if data exists
        if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
            print("\n⚠️  No training data found!")
            print(f"   Please add images to: {train_dir}")
            print("   Create a folder for each class (e.g., train/cats/, train/dogs/)")
            self._create_sample_data()
            return None, None
        
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        self.classes = train_dataset.classes
        print(f"\n📊 Found {len(self.classes)} classes: {self.classes}")
        print(f"   Training images: {len(train_dataset)}")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config["batch_size"],
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        # Validation set (optional)
        val_loader = None
        if os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0:
            val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            print(f"   Validation images: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _create_sample_data(self):
        """Create sample data structure"""
        train_dir = os.path.join(self.config["data_dir"], "train")
        sample_classes = ["class_a", "class_b", "class_c"]
        for cls in sample_classes:
            os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        print(f"\n   Created sample folders: {sample_classes}")
        print("   Add your images and run again!")
    
    def build_model(self, num_classes):
        """Build the model architecture"""
        model_type = self.config["model_type"]
        
        if model_type == "custom":
            model = CustomCNN(num_classes)
        elif model_type == "resnet18":
            model = models.resnet18(pretrained=self.config["pretrained"])
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == "resnet34":
            model = models.resnet34(pretrained=self.config["pretrained"])
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == "resnet50":
            model = models.resnet50(pretrained=self.config["pretrained"])
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == "efficientnet":
            model = models.efficientnet_b0(pretrained=self.config["pretrained"])
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Freeze backbone if using transfer learning
        if self.config["pretrained"] and self.config["freeze_backbone"] and model_type != "custom":
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze final layer
            if hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
        
        self.model = model.to(self.device)
        print(f"\n🏗️  Model: {model_type}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return model
    
    def unfreeze_model(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("🔓 Unfreezing all layers for fine-tuning")
    
    def train(self):
        """Main training loop"""
        train_loader, val_loader = self.setup_data()
        if train_loader is None:
            return
        
        num_classes = len(self.classes)
        self.build_model(num_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config["learning_rate"],
            weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["epochs"]
        )
        
        print(f"\n🚀 Starting training for {self.config['epochs']} epochs...\n")
        best_acc = 0.0
        
        for epoch in range(self.config["epochs"]):
            # Unfreeze backbone after specified epoch
            if epoch == self.config["unfreeze_epoch"] and self.config["freeze_backbone"]:
                self.unfreeze_model()
                optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.config["learning_rate"] * 0.1,
                    weight_decay=0.01
                )
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    "loss": f"{running_loss/total:.4f}",
                    "acc": f"{100.*correct/total:.2f}%"
                })
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader, criterion)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save_model("best_model.pth")
            
            scheduler.step()
        
        print(f"\n✅ Training complete! Best accuracy: {best_acc:.2f}%")
        self.save_model("final_model.pth")
        self.plot_history()
    
    def evaluate(self, data_loader, criterion):
        """Evaluate model on data loader"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(data_loader), 100. * correct / total
    
    def save_model(self, filename):
        """Save model and metadata"""
        os.makedirs(self.config["model_dir"], exist_ok=True)
        path = os.path.join(self.config["model_dir"], filename)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "classes": self.classes,
            "config": self.config,
            "history": self.history
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, filename):
        """Load a trained model"""
        path = os.path.join(self.config["model_dir"], filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.classes = checkpoint["classes"]
        self.config = checkpoint["config"]
        self.build_model(len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"📂 Model loaded from {path}")
    
    def predict(self, image_path):
        """Predict class for a single image"""
        if self.model is None:
            print("⚠️  No model loaded! Train or load a model first.")
            return None
        
        transform = transforms.Compose([
            transforms.Resize(self.config["image_size"] + 32),
            transforms.CenterCrop(self.config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = probabilities.argmax().item()
        
        result = {
            "class": self.classes[predicted_idx],
            "confidence": probabilities[predicted_idx].item(),
            "all_probabilities": {
                cls: prob.item() 
                for cls, prob in zip(self.classes, probabilities)
            }
        }
        return result
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history["train_loss"], label="Train")
        if self.history["val_loss"]:
            axes[0].plot(self.history["val_loss"], label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        
        axes[1].plot(self.history["train_acc"], label="Train")
        if self.history["val_acc"]:
            axes[1].plot(self.history["val_acc"], label="Val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.config["model_dir"], "training_history.png")
        plt.savefig(plot_path)
        print(f"📈 Training history saved to {plot_path}")
        plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("🖼️  IMAGE CLASSIFIER - CNN with PyTorch")
    print("=" * 60)
    
    classifier = ImageClassifier()
    
    # Train the model
    classifier.train()
    
    # Example: Load model and predict
    # classifier.load_model("best_model.pth")
    # result = classifier.predict("path/to/image.jpg")
    # print(result)
