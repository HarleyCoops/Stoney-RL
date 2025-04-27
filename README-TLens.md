# 🩻 TransformerLens Integration for Stoney-RL

This document explains how to integrate [TransformerLens](https://transformerlensorg.github.io/TransformerLens/) into the Stoney-RL pipeline to enable mechanistic interpretability for low-resource language reasoning tasks. TransformerLens provides tools to visualize and analyze the internals of transformer-based language models, helping researchers understand how models process and reason through linguistic data.

## Why TransformerLens?

TransformerLens offers critical insights into model behavior:

- **Attention Pattern Visualization**: See where your model focuses during reasoning tasks
- **Neuron Activation Analysis**: Identify which specific neurons activate for Stoney Nakoda language elements
- **Circuit Discovery**: Map out reasoning pathways through the model
- **Logit Attribution**: Understand how different model components contribute to predictions
- **Weights & Biases Integration**: All visualizations can be logged to W&B for experiment tracking

## Installation

Add TransformerLens and visualization dependencies to your environment:

```bash
pip install transformer-lens[wandb] plotly==5.*
```

This installs the core TransformerLens package plus the extras needed for Weights & Biases integration.

## Integration Options

There are two main approaches for integrating TransformerLens with your Stoney-RL pipeline:

1. **Live Training Instrumentation**: Monitor internals during training
2. **Post-hoc Analysis**: Analyze trained checkpoints after runs complete

## 1. Live Training Instrumentation

For real-time monitoring during training, create a utility wrapper (`tlens_hook.py`):

```python
# tlens_hook.py
import torch
import wandb
from transformer_lens import HookedTransformer
from contextlib import contextmanager

class TLensTracer:
    def __init__(self, model, run=None, every_n_steps=100):
        """Initialize the TransformerLens tracer.
        
        Args:
            model: The model being trained
            run: wandb.Run object for logging
            every_n_steps: Log visualizations every N steps
        """
        self.original_model = model
        self.run = run or wandb.run
        self.every_n_steps = every_n_steps
        self.step_counter = 0
        self.hooks = []
        
    def __enter__(self):
        # Convert to HookedTransformer format if possible, or wrap
        # This depends on your specific model architecture
        # For PEFT/LoRA models, you might need special handling
        self.tlens_model = HookedTransformer.from_pretrained(
            model_name=self.original_model.config._name_or_path,
            revision=None,
            fold_ln=False,
            center_unembed=False,
            center_writing_weights=False
        )
        
        # Register attention pattern hook
        def log_attention(attn_pattern, hook):
            if self.step_counter % self.every_n_steps == 0:
                # Log attention heatmaps to W&B
                # Take first example from batch for visualization
                head_idx = 0  # Can cycle through different heads
                layer_idx = hook.layer_idx
                
                fig = self.tlens_model.visualization.attention.attention_pattern_to_plot(
                    attention=attn_pattern[0, head_idx].detach().cpu(),
                    tokens=["[Sample tokens would be here]"],  # Would need actual tokens
                )
                
                self.run.log({
                    f"attention/layer_{layer_idx}_head_{head_idx}": wandb.Image(fig)
                })
            
        # Add hooks to appropriate layers
        for layer_idx in range(self.tlens_model.cfg.n_layers):
            hook = self.tlens_model.blocks[layer_idx].attn.hook_pattern
            self.hooks.append(hook.add_hook(log_attention))
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up by removing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
```

Integrate with your existing training loop:

```python
# In your training script
from tlens_hook import TLensTracer
import wandb

# Initialize wandb run
wandb_run = wandb.init(project="stoney-rl", name="lora_sft_r1zero_with_tlens")

# Load and prepare your model as usual
# ...

# Wrap your training loop with the TLensTracer
with TLensTracer(model, run=wandb_run, every_n_steps=100):
    trainer.train()  # Your SFTTrainer or GRPOTrainer call
```

## 2. Post-hoc Analysis

For deeper analysis after training, create an analysis script (`analyze_with_tlens.py`):

```python
# analyze_with_tlens.py
import argparse
import torch
import wandb
from pathlib import Path
from transformer_lens import HookedTransformer
import plotly.express as px

def main():
    parser = argparse.ArgumentParser(description="Analyze a trained model with TransformerLens")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--sample_text", type=str, default="[Sample Stoney text]", help="Prompt to analyze")
    parser.add_argument("--wandb_project", type=str, default="stoney-rl", help="W&B project name")
    parser.add_argument("--run_name", type=str, default="tlens_analysis", help="W&B run name")
    args = parser.parse_args()
    
    # Initialize W&B
    run = wandb.init(project=args.wandb_project, name=args.run_name)
    
    # Load model into TransformerLens format
    model = HookedTransformer.from_pretrained(
        model_name=args.model_path,
        revision=None,
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False
    )
    
    # Tokenize sample text
    tokens = model.tokenizer.encode(args.sample_text, return_tensors="pt")
    
    # Run model with hooks to capture internal activations
    logits, cache = model.run_with_cache(tokens)
    
    # Generate attention visualizations
    for layer_idx in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            attn_pattern = cache["pattern", layer_idx][0, head_idx].detach().cpu()
            tokenized_text = model.tokenizer.convert_ids_to_tokens(tokens[0])
            
            fig = model.visualization.attention.attention_pattern_to_plot(
                attention=attn_pattern,
                tokens=tokenized_text
            )
            
            # Log to W&B
            run.log({
                f"attention/layer_{layer_idx}_head_{head_idx}": wandb.Image(fig)
            })
    
    # Create interactive attention circuit visualization
    # Note: This requires additional HTML/JavaScript rendering
    # Simplified example:
    html_content = model.visualization.circuitsvis.attention_heads(
        tokens=tokenized_text,
        attention=cache["pattern"]
    )
    
    # Log HTML to W&B
    run.log({"circuits/attention_heads": wandb.Html(html_content)})
    
    # Create and log neuron activation heatmap
    mlp_acts = cache["mlp_out", 0].detach().cpu()
    fig = px.imshow(
        mlp_acts[0, :, :100].numpy(),  # First 100 neurons of first layer
        labels=dict(x="Neuron Index", y="Token Position"),
        title="MLP Neuron Activations - Layer 0"
    )
    run.log({"activations/mlp_layer_0": wandb.Image(fig)})
    
    # For comprehensive analysis:
    # - Log attention head influence scores
    # - Perform logit attribution analysis
    # - Identify key neurons for specific tokens
    # See TransformerLens docs for more advanced techniques
    
    run.finish()

if __name__ == "__main__":
    main()
```

Run the analysis on your trained checkpoint:

```bash
python analyze_with_tlens.py --model_path stoney_lora_merged --sample_text "dîrıtʼa ... [sample Stoney Nakoda text]"
```

## Viewing Results in Weights & Biases

Both approaches log visualizations to your W&B dashboard:

1. **Live Training View**: Monitor attention patterns evolving as training progresses
   - Find under the "Media" tab → "Images" section
   - Filter by "attention/" prefix in the sidebar

2. **Post-hoc Analysis Results**: Comprehensive, interactive visualizations
   - Find under "Media" → "Images" for static visualizations
   - Find under "Media" → "HTML" for interactive circuit diagrams
   - Artifacts section contains saved interactive reports

## Example: Analyzing Stoney Linguistic Features

TransformerLens is particularly valuable for examining how your model processes Stoney Nakoda linguistic features:

```python
# Example code snippet for linguistic feature analysis
from transformer_lens.utils import get_act_name

def analyze_linguistic_feature(model, text_with_feature, text_without_feature):
    """Compare activations between texts with and without a specific linguistic feature."""
    tokens_with = model.tokenizer.encode(text_with_feature, return_tensors="pt")
    tokens_without = model.tokenizer.encode(text_without_feature, return_tensors="pt")
    
    _, cache_with = model.run_with_cache(tokens_with)
    _, cache_without = model.run_with_cache(tokens_without)
    
    # Examine MLP activations at each layer
    for layer in range(model.cfg.n_layers):
        act_name = get_act_name("mlp_out", layer)
        
        # Compute activation difference
        act_diff = (
            cache_with[act_name][0].mean(dim=0) - 
            cache_without[act_name][0].mean(dim=0)
        )
        
        # Find top neurons with largest differential response
        top_neurons = torch.topk(act_diff.abs(), k=10)
        
        print(f"Layer {layer} - Top neurons responding to linguistic feature:")
        for i, (idx, val) in enumerate(zip(top_neurons.indices, top_neurons.values)):
            print(f"  {i+1}. Neuron {idx}: {val.item():.4f}")
```

## Further Resources

- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)
- [Anthropic's "Mechanistic Interpretability" Primer](https://transformer-circuits.pub/)
- [Weights & Biases Integration Guide](https://docs.wandb.ai/guides/integrations)

## Requirements

The integration requires:

- Python 3.8+
- PyTorch 1.12+
- transformer-lens 1.12+
- plotly 5.x
- wandb

## Troubleshooting

- **Memory Issues**: For large models, you may need to selectively cache only specific attention layers and MLP blocks.
- **Visualization Errors**: Ensure plotly and kaleido are properly installed for figure rendering.
- **W&B Connection**: Verify your W&B API key is set correctly via environment variable or `wandb login`.
