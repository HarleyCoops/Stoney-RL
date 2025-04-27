import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='LoRA fine-tuning for Stoney-RL')
parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Zero-7B", help='Base model to use')
parser.add_argument('--batch_size', type=int, default=4, help='Per device batch size')
parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
args = parser.parse_args()

# Initialize W&B
run_name = f"lora_sft_{args.model.split('/')[-1]}"
wandb_run = wandb.init(
    project="stoney-rl", 
    name=run_name, 
    config={
        "base_model": args.model, 
        "lora_rank": 16,
        "batch_size": args.batch_size,
        "multi_gpu": args.multi_gpu,
        "num_gpus": args.num_gpus
    }
)

# Load and prepare dataset
print("Loading dataset from HuggingFace Hub...")
ds = load_dataset("HarleyCooper/StoneyNakoda45k", split="train")

def flatten(triple):
    prompt = f"<system>{triple[0]['content']}\n<user>{triple[1]['content']}\n<assistant>"
    return {"prompt": prompt, "target": triple[2]["content"]}

train_ds = ds.map(flatten, remove_columns=ds.column_names)
print(f"Dataset prepared with {len(train_ds)} examples")

# Load model and tokenizer
print(f"Loading model {args.model}...")
base_model = args.model
tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model, 
    torch_dtype="auto", 
    device_map="auto" if args.multi_gpu else None
)

# Configure LoRA
print("Configuring LoRA...")
lora_cfg = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.05,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# Configure training arguments
print("Setting up training configuration...")
sft_args = TrainingArguments(
    output_dir="stoney_lora_ckpt",
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    save_strategy="epoch",
    report_to="wandb",
    run_name=wandb_run.name
)

# Initialize trainer and start training
print("Initializing trainer...")
trainer = SFTTrainer(
    model=model, 
    args=sft_args,
    train_dataset=train_ds,
    dataset_text_field="prompt",
    tokenizer=tok,
    max_seq_length=512
)

print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained("stoney_lora_merged")
tok.save_pretrained("stoney_lora_merged")

print("Training complete!") 