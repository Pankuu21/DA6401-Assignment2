import wandb
import yaml
import multiprocessing
import os
from Q2_Training_Hyperparameter_Tuning import train_model

def load_sweep_config(config_path='configs/sweep_config.yaml'):
    """Load sweep configuration from YAML file"""
    try:
        with open(config_path, 'r') as config_file:
            return yaml.load(config_file, Loader=yaml.SafeLoader)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        raise

def initialize_wandb(api_key='768b934ca374980c7a99de19dbc63ad89ec3b865'):
    """Initialize WandB with the provided API key"""
    try:
        wandb.login(key=api_key)
        print("Successfully logged in to Weights & Biases")
        return True
    except Exception as e:
        print(f"Failed to log in to Weights & Biases: {e}")
        return False

def run_sweep():
    """Configure and run the hyperparameter sweep"""
    # Load sweep configuration
    sweep_config = load_sweep_config()
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="DA6401-Assignment2-CNN"
    )
    
    # Start the sweep agent
    print(f"Starting sweep with ID: {sweep_id}")
    wandb.agent(sweep_id, train_model)
    
    return sweep_id

if __name__ == "__main__":
    # Initialize multiprocessing support for Windows
    multiprocessing.freeze_support()
    
    # Run the sweep pipeline
    if initialize_wandb():
        try:
            sweep_id = run_sweep()
            print(f"Sweep completed successfully. Sweep ID: {sweep_id}")
        except Exception as e:
            print(f"Error during sweep execution: {e}")
    else:
        print("Failed to initialize WandB. Exiting...")
