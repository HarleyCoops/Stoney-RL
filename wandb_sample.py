# This script demonstrates how to use Weights & Biases (wandb) to track machine learning experiments.
# It simulates a training loop and logs metrics to the wandb dashboard for visualization and analysis.

import random  # Import the random module to simulate metric values.
import time  # Import the time module for sleep

import wandb  # Import the wandb library for experiment tracking.

def main():
    """
    Simple script to test Weights & Biases integration.
    This will create a test run and log some dummy metrics.
    """
    print("Testing Weights & Biases connection...")
    
    # Initialize W&B run
    run = wandb.init(
        project="stoney-rl",
        name="connectivity_test",
        config={
            "test_param1": 42,
            "test_param2": "hello",
            "environment": "hyperbolic-labs"
        }
    )
    
    # Simulate some training metrics
    print("Logging test metrics...")
    for i in range(10):
        metrics = {
            "loss": 1.0 - i * 0.1 + random.random() * 0.05,
            "accuracy": i * 10 + random.random() * 5
        }
        wandb.log(metrics)
        time.sleep(0.5)  # Just to space out the logging
    
    # Create a simple table
    table = wandb.Table(columns=["step", "value"])
    for i in range(5):
        table.add_data(i, i * 10)
    wandb.log({"test_table": table})
    
    # Finish the run
    wandb.finish()
    
    print("W&B test complete! Check your dashboard to confirm the test run was created.")
    print("Dashboard URL: https://wandb.ai/<username>/stoney-rl")
    

if __name__ == "__main__":
    main()