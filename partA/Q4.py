import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import wandb
from Q1_Flexible_CNN_Model import FlexibleCNN

class InaturalistTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = None
        self.all_images = []
        self.all_labels = []
        self.all_preds = []
        
    def setup_data(self):
        """Prepare data loaders for training, validation and testing"""
        # Define transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Setup train/val split
        train_dir = "C:/Users/pankaj/Desktop/dl/DL_assignment_2/DA6401-Assignment2/data/inaturalist_12K/train"
        full_train_dataset = datasets.ImageFolder(train_dir, transform=None)
        targets = [label for _, label in full_train_dataset.samples]
        
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))
        
        train_dataset = Subset(datasets.ImageFolder(train_dir, transform=train_transform), train_idx)
        val_dataset = Subset(datasets.ImageFolder(train_dir, transform=val_test_transform), val_idx)
        
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
        
        # Setup test data
        test_dir = "C:/Users/pankaj/Desktop/dl/DL_assignment_2/DA6401-Assignment2/data/inaturalist_12K/val"
        test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=False, 
            num_workers=0
        )
        
        self.class_names = test_dataset.classes
        
    def setup_model(self):
        """Initialize model, optimizer and loss function"""
        self.model = FlexibleCNN(
            conv_filters=self.config["conv_filters"],
            kernel_sizes=self.config["kernel_sizes"],
            activation=self.config["activation"],
            dense_neurons=self.config["dense_neurons"],
            dropout=self.config["dropout"],
            use_batchnorm=self.config["use_batchnorm"]
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass and optimization
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        return train_loss / train_total, train_correct / train_total
        
    def validate(self):
        """Run validation"""
        self.model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        return val_loss / val_total, val_correct / val_total
    
    def test(self):
        """Evaluate on test set and collect predictions"""
        self.model.eval()
        correct, total = 0, 0
        self.all_images, self.all_labels, self.all_preds = [], [], []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                # Track statistics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Save for visualization
                self.all_images.extend(images.cpu())
                self.all_labels.extend(labels.cpu().numpy())
                self.all_preds.extend(predicted.cpu().numpy())
                
        test_acc = correct / total
        return test_acc
    
    def visualize_predictions(self, num_samples=30):
        """Visualize model predictions"""
        plt.figure(figsize=(15, 20))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        for idx in range(num_samples):
            plt.subplot(10, 3, idx+1)
            img = self.all_images[idx].permute(1, 2, 0).numpy()
            img = std * img + mean  # Denormalize
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            
            color = 'green' if self.all_labels[idx] == self.all_preds[idx] else 'red'
            plt.title(f"True: {self.class_names[self.all_labels[idx]]}\n"
                      f"Pred: {self.class_names[self.all_preds[idx]]}", color=color)
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig('test_predictions.png')
        plt.show()
        
        # Log to W&B
        wandb.log({"test_predictions_grid": wandb.Image("test_predictions.png")})
    
    def train_and_evaluate(self):
        """Main training loop"""
        # Setup
        self.setup_data()
        self.setup_model()
        
        # Training loop
        for epoch in range(self.config["epochs"]):
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            # Log metrics
            wandb.log({
                "epoch": epoch+1,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
            
            print(f"Epoch {epoch+1}/{self.config['epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Test evaluation
        test_acc = self.test()
        wandb.log({"test_acc": test_acc})
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Visualize results
        self.visualize_predictions()


def main():
    # Configuration
    config = {
        "conv_filters": [16, 32, 64, 64, 128],
        "kernel_sizes": [3, 3, 3, 5, 5],
        "activation": "gelu",
        "dense_neurons": 512,
        "dropout": 0.2,
        "use_batchnorm": True,
        "learning_rate": 0.0003447,
        "batch_size": 16,
        "epochs": 15
    }
    
    # Initialize W&B
    wandb.init(project="DA6401-Assignment2-CNN", config=config, name="gelu-16-16-512")
    
    # Train and evaluate
    trainer = InaturalistTrainer(config)
    trainer.train_and_evaluate()


if __name__ == "__main__":
    main()