import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import wandb
import os
from Q1_Flexible_CNN_Model import FlexibleCNN
import multiprocessing

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_acc = 0.0
        self.model = None
        self.criterion = None
        self.optimizer = None
        
    def prepare_data(self):
        # Define transforms based on augmentation flag
        train_transform = self._get_train_transform()
        val_transform = self._get_val_transform()
        
        # Load dataset with correct path
        train_dir = 'C:/Users/pankaj/Desktop/dl/DL_assignment_2/DA6401-Assignment2/data/inaturalist_12K/train'
        full_train_dataset = datasets.ImageFolder(train_dir, transform=None)
        targets = [label for _, label in full_train_dataset.samples]
        
        # Create stratified split (20% validation)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))
        
        # Create datasets with transforms
        train_dataset = Subset(
            datasets.ImageFolder(train_dir, transform=train_transform), 
            train_idx
        )
        val_dataset = Subset(
            datasets.ImageFolder(train_dir, transform=val_transform), 
            val_idx
        )
        
        # Verify class distribution in validation set
        class_counts = {i: 0 for i in range(10)}
        for idx in val_idx:
            _, label = full_train_dataset.samples[idx]
            class_counts[label] += 1
        print("Validation set class distribution:", class_counts)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=False,
            num_workers=0
        )
    
    def _get_train_transform(self):
        if self.config["use_augmentation"]:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _get_val_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup_model(self):
        # Initialize model
        self.model = FlexibleCNN(
            conv_filters=self.config["conv_filters"],
            kernel_sizes=self.config["kernel_sizes"],
            activation=self.config["activation"],
            dense_neurons=self.config["dense_neurons"],
            dropout=self.config.get("dropout", 0.2),
            use_batchnorm=self.config.get("use_batchnorm", True)
        ).to(self.device)
        
        # Loss function and optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        
        # Log model architecture
        wandb.watch(self.model, log="all")
    
    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        return train_loss, train_acc
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Save best model
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(self.model.state_dict(), "best_model.pth")
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        return val_loss, val_acc
    
    def run(self):
        # Create a descriptive run name
        run_name = f"{self.config['activation']}-{self.config['batch_size']}-{self.config['conv_filters'][0]}-{self.config['dense_neurons']}"
        
        with wandb.init(config=self.config, name=run_name) as run:
            # Use the wandb config object
            self.config = dict(wandb.config)
            
            print(f"Using device: {self.device}")
            
            # Setup data and model
            self.prepare_data()
            self.setup_model()
            
            # Handle epochs parameter - could be a list or a single value
            num_epochs = self.config["epochs"]
            if isinstance(num_epochs, list):
                num_epochs = num_epochs[0]  # Take first value if it's a list
            
            # Training loop
            for epoch in range(num_epochs):
                # Train and validate
                train_loss, train_acc = self.train_epoch(epoch, num_epochs)
                val_loss, val_acc = self.validate()
                
                # Log metrics to wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
                
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            wandb.save("best_model.pth")
            return self.best_val_acc

def run_training(config):
    trainer = ModelTrainer(config)
    return trainer.run()

if __name__ == "__main__":
    # This is critical for Windows multiprocessing
    multiprocessing.freeze_support()
    
    # Initialize wandb
    wandb.login()
    
    # Define configuration
    config = {
        "conv_filters": [16, 32, 64, 64, 128],
        "kernel_sizes": [3, 3, 3, 3, 3],
        "activation": "relu",
        "dense_neurons": 256,
        "dropout": 0.2,
        "use_batchnorm": True,
        "use_augmentation": True,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 15
    }
    
    # Run training
    run_training(config)