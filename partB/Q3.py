import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import wandb
import multiprocessing

class TransferLearningTrainer:
    def __init__(self, config):
        """Initialize the trainer with configuration parameters"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_acc = 0.0
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.class_names = None
        
    def prepare_data(self):
        """Set up data loaders for training, validation and testing"""
        # Define transformations
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load datasets with paths
        train_dir = 'C:/Users/pankaj/Desktop/dl/DL_assignment_2/DA6401-Assignment2/data/inaturalist_12K/train'
        test_dir = 'C:/Users/pankaj/Desktop/dl/DL_assignment_2/DA6401-Assignment2/data/inaturalist_12K/val'
        
        # Create train/validation split
        full_train_dataset = datasets.ImageFolder(train_dir, transform=None)
        targets = [label for _, label in full_train_dataset.samples]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))
        
        # Create dataset objects
        train_dataset = Subset(datasets.ImageFolder(train_dir, transform=preprocess), train_idx)
        val_dataset = Subset(datasets.ImageFolder(train_dir, transform=preprocess), val_idx)
        test_dataset = datasets.ImageFolder(test_dir, transform=preprocess)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        # Store class names for visualization
        self.class_names = test_dataset.classes
        
    def setup_model(self):
        """Initialize pre-trained ResNet50 and modify for transfer learning"""
        # Load pre-trained model
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Replace final classification layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10)  # 10 classes for iNaturalist
        
        # Freeze all layers except the last
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
            
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        # Calculate metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        return train_loss, train_acc
        
    def validate(self):
        """Run validation on the validation set"""
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
                
        # Calculate metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Save best model
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(self.model.state_dict(), "best_resnet_model.pth")
            print(f"Model saved with validation accuracy: {val_acc:.4f}")
            
        return val_loss, val_acc
    
    def evaluate_test_set(self):
        """Evaluate on test set and collect predictions for visualization"""
        # Load best model
        self.model.load_state_dict(torch.load("best_resnet_model.pth"))
        self.model.eval()
        
        correct, total = 0, 0
        all_images, all_labels, all_preds = [], [], []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                # Track statistics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for visualization
                all_images.extend(images.cpu())
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
        test_acc = correct / total
        print(f"Test Accuracy: {test_acc:.4f}")
        wandb.log({"test_acc": test_acc})
        
        return test_acc, all_images, all_labels, all_preds
    
    def log_prediction_grid(self, all_images, all_labels, all_preds):
        """Create and log visualization grid to wandb"""
        columns = ["image", "true_label", "pred_label", "correct"]
        table = wandb.Table(columns=columns)
        
        # For denormalizing images
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Add samples to table
        for idx in range(min(30, len(all_images))):
            img = all_images[idx].permute(1, 2, 0).numpy()
            img = std * img + mean
            img = np.clip(img, 0, 1)
            correct_pred = (all_labels[idx] == all_preds[idx])
            
            table.add_data(
                wandb.Image(img, caption=f"True: {self.class_names[all_labels[idx]]}, Pred: {self.class_names[all_preds[idx]]}"),
                self.class_names[all_labels[idx]],
                self.class_names[all_preds[idx]],
                correct_pred
            )
            
        wandb.log({"Test Prediction Grid": table})
        
    def train_and_evaluate(self):
        """Main training and evaluation loop"""
        print(f"Using device: {self.device}")
        
        # Setup
        self.prepare_data()
        self.setup_model()
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Log metrics
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
            
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                 f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
        # Test evaluation
        test_acc, all_images, all_labels, all_preds = self.evaluate_test_set()
        
        # Log visualizations
        self.log_prediction_grid(all_images, all_labels, all_preds)


def main():
    # Initialize wandb and config
    wandb.init(project="DA6401-Assignment2", name="gelu-16-16-512")
    config = wandb.config
    config.activation = "gelu"
    config.batch_size = 16
    config.conv_filters = [16,32,64,64,128]
    config.dense_neurons = 512
    config.epochs = 15
    config.kernel_sizes = [3,3,3,5,5]
    config.learning_rate = 0.0003447
    
    # Create and run the trainer
    trainer = TransferLearningTrainer(config)
    trainer.train_and_evaluate()
    
    # Finish wandb run
    wandb.finish()

# Critical for Windows multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()