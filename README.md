# DA6401 Assignment 2

**Name:** Pankaj Badgujar  
**Roll No.:** CS24M029
**WandB Project:** DA6401-Assignment2


Report Link :https://api.wandb.ai/links/cs24m029-indian-institute-of-technology-madras/sq8pfipx
---


## Setup Instructions

### 1. Dependencies

Install required packages:
```
pip install torch torchvision wandb scikit-learn matplotlib numpy pyyaml
```

### 2. Dataset

- Download and extract the iNaturalist dataset.
- Place it in `data/inaturalist_12K/` with `train` and `val` folders.

### 3. Weights & Biases

- Create a wandb account and get your API key.
- Login in your terminal or in code:
  ```
  import wandb
  wandb.login()
  ```

---

## Part A: Training a Flexible CNN from Scratch

### Key Files

- `Q1_Flexible_CNN_Model.py`: Flexible CNN architecture.
- `Q2_Training_Hyperparameter_Tuning.py`: Hyperparameter sweep logic and training.
- `Q2_Run_Sweep.py`: Script to launch and run sweeps using W&B.
- `Q3.py`: Analysis and observations from sweeps.
- `Q4.py` / `Q4.ipynb`: Test set evaluation and prediction visualization.
- `configs/sweep_config.yaml`: Sweep configuration for W&B.

### Usage

1. **Run Hyperparameter Sweep**
   ```
   python partA/Q2_Run_Sweep.py
   ```
   (This will use the sweep config in `configs/sweep_config.yaml`.)

2. **Evaluate Best Model**
   ```
   python partA/Q4.py
   ```
   or open `Q4.ipynb` and run all cells.

---

## Part B: Fine-tuning a Pre-trained Model

### Key File

- `Q3.py`: Fine-tune ResNet50 (pretrained on ImageNet) on iNaturalist, evaluate, and log results to wandb.

### Usage

1. **Fine-tune and Evaluate**
   ```
   python partB/Q3.py
   ```

---

## Results & Visualization

- All training/validation/test metrics and prediction grids are logged to Weights & Biases.
- You can view your runs and plots at your W&B project URL.

---



## References

- Assignment PDF: DA6401-Assignment-2-_-DA6401-Weights-Biases.pdf
- [W&B Documentation](https://docs.wandb.ai/)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---




