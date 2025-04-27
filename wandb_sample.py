# This script demonstrates how to use Weights & Biases (wandb) to track machine learning experiments.
# It simulates a training loop and logs metrics to the wandb dashboard for visualization and analysis.

import random  # Import the random module to simulate metric values.

import wandb  # Import the wandb library for experiment tracking.

# Start a new wandb run to track this script.
# The run object manages the connection to the wandb server and handles logging.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="christian-cooper-us",  # Replace with your actual W&B team or username.
    # Set the wandb project where this run will be logged.
    project="stoney-rl",   # Replace with your actual W&B project name.
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,      # Example hyperparameter: learning rate for the optimizer.
        "architecture": "CNN",     # Example metadata: model architecture type.
        "dataset": "CIFAR-100",    # Example metadata: dataset used for training.
        "epochs": 10,               # Example hyperparameter: number of training epochs.
    },
)

# Simulate a training loop for demonstration purposes.
epochs = 10  # Total number of epochs to simulate.
offset = random.random() / 5  # Add a small random offset to make the metrics less predictable.
for epoch in range(2, epochs):  # Start from epoch 2 for demonstration; typically would start from 0 or 1.
    # Simulate an accuracy value that increases with each epoch.
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    # Simulate a loss value that decreases with each epoch.
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb for this epoch.
    # The keys in the dictionary ("acc" and "loss") will appear as metric names in the wandb dashboard.
    run.log({"acc": acc, "loss": loss})

# Finish the run and upload any remaining data to the wandb server.
# This ensures all metrics and artifacts are saved and the run is properly closed.
run.finish()