import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sacrebleu import corpus_chrf
import argparse
import torch
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate Stoney-RL model')
parser.add_argument('--model_path', type=str, default="stoney_rl_grpo", help='Path to the trained model')
parser.add_argument('--num_samples', type=int, default=400, help='Number of samples to evaluate')
parser.add_argument('--run_comet', action='store_true', help='Run COMET evaluation (requires more dependencies)')
parser.add_argument('--run_mauve', action='store_true', help='Run MAUVE evaluation (requires more dependencies)')
args = parser.parse_args()

# Initialize W&B
wandb_run = wandb.init(
    project="stoney-rl",
    name=f"eval_{os.path.basename(args.model_path)}",
    config={
        "model_path": args.model_path,
        "num_samples": args.num_samples,
        "run_comet": args.run_comet,
        "run_mauve": args.run_mauve
    }
)

# Load dataset
print("Loading dataset...")
ds = load_dataset("HarleyCooper/synthetic_stoney_data", split="train")

def flatten(triple):
    prompt = f"<system>{triple[0]['content']}\n<user>{triple[1]['content']}\n<assistant>"
    return {"prompt": prompt, "target": triple[2]["content"]}

eval_ds = ds.map(flatten, remove_columns=ds.column_names)
print(f"Dataset loaded with {len(eval_ds)} examples")

# Select a subset for evaluation
eval_ds = eval_ds.select(range(min(args.num_samples, len(eval_ds))))
print(f"Using {len(eval_ds)} examples for evaluation")

# Load model and tokenizer
print(f"Loading model from {args.model_path}...")
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# Generate predictions
print("Generating predictions...")
refs = [ex["target"] for ex in eval_ds]
prompts = [ex["prompt"] for ex in eval_ds]
cands = []

for i, p in enumerate(prompts):
    print(f"Processing example {i+1}/{len(prompts)}", end="\r")
    
    inputs = tokenizer(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_return_sequences=1
        )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant response
    assistant_response = output_text.split("<assistant>")[-1].strip()
    cands.append(assistant_response)

print("\nEvaluation metrics:")

# Calculate chrF++
print("Calculating chrF++...")
chrf_score = corpus_chrf(cands, [refs]).score
print(f"chrF++: {chrf_score}")
wandb.log({"chrf": chrf_score})

# Calculate COMET if requested
if args.run_comet:
    try:
        print("Calculating COMET...")
        from comet import download_model, load_from_checkpoint
        
        ckpt = download_model("Unbabel/wmt22-cometkiwi-da")
        comet_score = load_from_checkpoint(ckpt).predict(cands, refs, batch_size=16)
        print(f"COMET: {comet_score}")
        wandb.log({"comet": comet_score})
    except Exception as e:
        print(f"Error calculating COMET: {e}")

# Calculate MAUVE if requested
if args.run_mauve:
    try:
        print("Calculating MAUVE...")
        import mauve
        
        mauve_score = mauve.compute_mauve(p_text=cands, q_text=refs).mauve
        print(f"MAUVE: {mauve_score}")
        wandb.log({"mauve": mauve_score})
    except Exception as e:
        print(f"Error calculating MAUVE: {e}")

# Log examples to W&B
examples_table = wandb.Table(columns=["prompt", "reference", "prediction"])
for i in range(min(10, len(cands))):
    examples_table.add_data(prompts[i], refs[i], cands[i])
wandb.log({"examples": examples_table})

print("Evaluation complete!")
wandb.finish() 