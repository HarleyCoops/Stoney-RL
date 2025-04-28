import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from dotenv import load_dotenv
import torch # Import torch to check available memory

# Load environment variables from .env file
load_dotenv() 

# --- Configuration ---
base_model = "microsoft/Phi-3-mini-4k-instruct" # Suitable for ~16GB VRAM
# dataset_name = "HarleyCooper/StoneyNakoda45k" # Incorrect dataset name
dataset_name = "HarleyCooper/synthetic_stoney_data" # Correct dataset name
lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"] # Check these module names for Phi-3
train_batch_size = 2 # Adjust based on GPU memory
gradient_accumulation_steps = 8 # Adjust based on GPU memory
learning_rate = 2e-4
num_train_epochs = 1
output_dir = "stoney_lora_phi3_local_ckpt"
save_dir = "stoney_lora_phi3_local_merged"
max_seq_length = 512
logging_steps = 25
wandb_project = "stoney-rl" # Ensure this matches your W&B project
run_name = "lora_sft_phi3_local"

# --- Dataset Loading and Preparation ---
print(f"Loading dataset '{dataset_name}' from HuggingFace Hub...")
ds = load_dataset(dataset_name, split="train", data_files="extracted_qa_pairs.jsonl")

def flatten(triple):
    # Assuming the structure is like [{'role':'system', 'content':'...'}, {'role':'user', 'content':'...'}, {'role':'assistant', 'content':'...'}]
    # Adjust indices/keys if the actual structure is different
    system_prompt = triple[0].get('content', '') # Handle potential missing system prompt
    user_prompt = triple[1].get('content', '')
    assistant_response = triple[2].get('content', '')
    
    # Format based on Phi-3's expected chat template (check model card if unsure)
    # Example format - adjust if needed:
    prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>"
    return {"prompt": prompt, "target": assistant_response}

train_ds = ds.map(flatten, remove_columns=ds.column_names)
print(f"Dataset prepared with {len(train_ds)} examples")

# --- W&B Initialization ---
print(f"Initializing Weights & Biases run: {run_name} in project: {wandb_project}")
wandb_run = wandb.init(
    project=wandb_project, 
    name=run_name, 
    config={
        "base_model": base_model, 
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "batch_size": train_batch_size,
        "gradient_accumulation": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "epochs": num_train_epochs,
        "max_seq_length": max_seq_length,
        "dataset": dataset_name
    }
)

# --- Model and Tokenizer Loading ---
print(f"Loading model '{base_model}'...")
tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
# Phi-3 might not have a default pad token, set it to eos_token
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    print("Set pad_token to eos_token")

# Check GPU memory before loading model
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU detected. Total Memory: {gpu_mem:.2f} GB")
else:
    print("No GPU detected. Training will run on CPU (slow).")
    
model = AutoModelForCausalLM.from_pretrained(
    base_model, 
    torch_dtype="auto", # Use auto for potential mixed precision
    device_map="auto", # Automatically distribute across available GPUs/CPU
    trust_remote_code=True # Needed for Phi-3
)
print("Model loaded.")

# --- LoRA Configuration ---
print("Configuring LoRA...")
# Verify target_modules for Phi-3. Common ones include 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'.
# You might need to inspect the model architecture if unsure. Sticking with the common ones for now.
lora_cfg = LoraConfig(
    r=lora_rank, 
    lora_alpha=lora_alpha, 
    lora_dropout=lora_dropout,
    target_modules=target_modules,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
print("LoRA configured.")
model.print_trainable_parameters()

# --- Training Configuration ---
print("Setting up training configuration...")
sft_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    fp16=True, # Enable mixed precision training if GPU supports it
    logging_steps=logging_steps,
    save_strategy="epoch",
    report_to="wandb",
    run_name=wandb_run.name,
    optim="paged_adamw_8bit" # Optimizer choice for memory efficiency
)

# --- Trainer Initialization and Training ---
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model, 
    args=sft_args,
    train_dataset=train_ds,
    dataset_text_field="prompt", # Use the 'prompt' field created by flatten
    # We are training on the prompt format including the start of the assistant response
    # The labels will be automatically generated by SFTTrainer up to max_seq_length
    tokenizer=tok,
    max_seq_length=max_seq_length,
    peft_config=lora_cfg # Pass LoRA config here as well
)

print("Starting training...")
trainer.train()
print("Training finished.")

# --- Save Model ---
print(f"Saving final LoRA adapter model to '{save_dir}'...")
# Save the LoRA adapter weights, not the merged model initially
trainer.save_model(save_dir) # Saves adapter_model.bin, adapter_config.json etc.
# model.save_pretrained(save_dir) # Use trainer.save_model for adapters
tok.save_pretrained(save_dir)
print("Model and tokenizer saved.")

wandb.finish()
print("Run finished and logged to W&B.") 