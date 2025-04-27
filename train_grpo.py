import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='GRPO training for Stoney-RL')
parser.add_argument('--lora_path', type=str, default="stoney_lora_merged", help='Path to LoRA fine-tuned model')
parser.add_argument('--kl_penalty', type=float, default=0.1, help='KL penalty for GRPO')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--group_size', type=int, default=16, help='Group size for GRPO')
parser.add_argument('--num_episodes', type=int, default=256, help='Number of episodes to train')
args = parser.parse_args()

# Initialize W&B
wandb_run = wandb.init(
    project="stoney-rl",
    name="grpo_rl_phase",
    config={
        "algo": "GRPO",
        "kl_penalty": args.kl_penalty,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "lora_path": args.lora_path,
        "num_episodes": args.num_episodes
    }
)

# Load dataset
print("Loading dataset...")
ds = load_dataset("HarleyCooper/StoneyNakoda45k", split="train")

def flatten(triple):
    prompt = f"<system>{triple[0]['content']}\n<user>{triple[1]['content']}\n<assistant>"
    return {"prompt": prompt, "target": triple[2]["content"]}

train_ds = ds.map(flatten, remove_columns=ds.column_names)
print(f"Dataset prepared with {len(train_ds)} examples")

# Load fine-tuned model
print(f"Loading model from {args.lora_path}...")
policy = AutoModelForCausalLM.from_pretrained(
    args.lora_path,
    torch_dtype="auto", 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.lora_path)

# Simple reward model function
def reward_fn(samples):
    # This is a placeholder reward function
    # In a real implementation, you would use a trained reward model
    print(f"Computing rewards for {len(samples)} samples")
    return [0.0 for _ in samples]

# Configure GRPO
print("Configuring GRPO trainer...")
grpo_cfg = GRPOConfig(
    batch_size=args.batch_size,
    group_size=args.group_size,
    learning_rate=2e-5,
    kl_penalty=args.kl_penalty,
    max_length=256,
    report_to="wandb",
    run_name=wandb_run.name
)

# Initialize GRPO trainer
print("Initializing GRPO trainer...")
grpo_trainer = GRPOTrainer(
    config=grpo_cfg,
    model=policy,
    tokenizer=tokenizer,
    dataset=train_ds,
    reward_fn=reward_fn
)

# Start training
print(f"Starting GRPO training for {args.num_episodes} episodes...")
grpo_trainer.train(num_episodes=args.num_episodes)

# Save model
print("Saving model...")
grpo_trainer.save_pretrained("stoney_rl_grpo")

print("GRPO training complete!") 