# Stoney-RL: Mechanistic Interpretability for Low-Resource Language Reasoning

## Overview
Stoney-RL is a research pipeline for exploring mechanistic interpretability in language models, with a focus on low-resource First Nations languages. The project leverages LoRA (Low-Rank Adaptation) fine-tuning, reward modeling, and GRPO (Group Regularized Policy Optimization) reinforcement learning to study how models reason through linguistic data, and how this reasoning differs from mathematical or high-resource language tasks.

## Key Features
- **LoRA Fine-Tuning:** Efficient parameter-efficient fine-tuning of large language models on reasoning-centric tasks.
- **Custom Reward Modeling:** Plug-and-play reward model stub for future multi-signal or human feedback integration.
- **GRPO RL Training:** Use of Hugging Face TRL's GRPOTrainer for reinforcement learning with custom rewards.
- **Mechanistic Interpretability:** Designed to probe and visualize attention mechanisms and reasoning steps in low-resource language contexts.
- **Weights & Biases Integration:** All experiments are tracked and logged for reproducibility and analysis.
- **Open, Synthetic Dataset:** Uses a synthetic Stoney Nakoda dataset, curated for linguistic reasoning and interpretability research.

## Project Structure
- `Instructions.txt`: Step-by-step playbook for reproducing the pipeline.
- `inspect_hf_dataset.py`: Script to inspect and validate the Hugging Face dataset.
- `wandb_sample.py`: Example script for logging metrics to Weights & Biases.
- `synthetic_stoney_data.jsonl`: The main dataset, uploaded to the Hugging Face Hub.

## Dataset
- **Source:** [HarleyCooper/synthetic-stoney-data](https://huggingface.co/datasets/HarleyCooper/synthetic-stoney-data)
- **Format:** Each entry contains a `question`, `answer`, and metadata fields. Designed for prompt/response training and evaluation.
- **Purpose:** Enables research on reasoning, attention, and interpretability in a low-resource, indigenous language context.

## Pipeline Steps
1. **Environment Setup**
   - Install dependencies (see `Instructions.txt`)
   - Configure Weights & Biases
2. **Dataset Inspection & Preparation**
   - Use `inspect_hf_dataset.py` to validate and explore the dataset
3. **Model Selection**
   - Choose a base model (e.g., DeepSeek-R1-Zero-7B, Phi-3-Mini-3.8B)
4. **LoRA Fine-Tuning**
   - Run supervised fine-tuning with LoRA adapters
   - Log all runs to W&B
5. **Reward Modeling (Optional)**
   - Implement or fine-tune a reward model for RL
6. **GRPO RL Training**
   - Reinforcement learning with custom reward function
7. **Evaluation**
   - Automatic evaluation with chrF, COMET, MAUVE, and W&B logging
8. **Hyperparameter Sweeps**
   - Use W&B sweeps for systematic search

## How to Run
1. **Clone the repo:**
   ```sh
   git clone https://github.com/HarleyCoops/Stoney-RL.git
   cd Stoney-RL
   ```
2. **Install dependencies:**
   See `Instructions.txt` for conda and pip setup.
3. **Inspect the dataset:**
   ```sh
   python inspect_hf_dataset.py
   ```
4. **Run LoRA fine-tuning:**
   Follow the code and instructions in `Instructions.txt`.
5. **Track experiments:**
   All runs are logged to Weights & Biases under the `stoney-rl` project.

## Distributed Training with Hyperbolic Labs

This project is configured for distributed training on Hyperbolic Labs cloud infrastructure. This approach offers:

- **Scalable GPU Access:** Access to high-memory NVIDIA GPUs (A100, H100) for faster training
- **Managed Environment:** Pre-configured CUDA environments with optimal drivers
- **Cost Effectiveness:** Pay-as-you-go compute versus fixed hardware costs
- **Parallelization:** Ability to run multiple experiments simultaneously

### Environment Requirements

When configuring your Hyperbolic Labs instance, ensure it meets these specifications:

- **Linux with CUDA 11.8+** (driver ≥ 525)
- **GPU Memory:**
  - 24+ GB VRAM for DeepSeek-R1-Zero-7B (A10, A100, H100)
  - 12+ GB VRAM for Phi-3-Mini-3.8B quantized models (T4, RTX 4000)
- **Storage:** Minimum 100GB SSD for model weights and dataset
- **RAM:** 32GB+ recommended

### Setup Process

1. **Initialize Hyperbolic Labs Instance:**
   ```sh
   # Clone repository to your Hyperbolic Labs instance
   git clone https://github.com/HarleyCoops/Stoney-RL.git
   cd Stoney-RL
   ```

2. **Configure Environment:**
   ```sh
   # Create and activate conda environment
   conda create -n stoney-rl python=3.11 -y
   conda activate stoney-rl
   
   # Install dependencies (CUDA drivers are pre-installed on Hyperbolic Labs)
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

3. **Verify W&B Connection:**
   ```sh
   # Set up Weights & Biases
   export WANDB_API_KEY="your_api_key"
   export WANDB_ENTITY="your_username"
   export WANDB_PROJECT="stoney-rl"
   
   # Test connection
   python wandb_sample.py
   ```

4. **Launch Training:**
   ```sh
   # Run LoRA fine-tuning
   python train_lora.py --model deepseek-ai/DeepSeek-R1-Zero-7B --batch_size 8
   
   # For multi-GPU training add:
   # --multi_gpu --num_gpus 4
   ```

5. **Monitor Progress:**
   Training progress can be monitored in real-time through your Weights & Biases dashboard at `https://wandb.ai/your_username/stoney-rl`

### Multi-node Training (Advanced)

For extremely large models or dataset sizes, multi-node training can be configured using Hyperbolic Labs' cluster:

```sh
# Example using torchrun for distributed training
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="master_ip" --master_port=1234 train_distributed.py
```

## Research Goals
- **Mechanistic Interp:** Visualize and analyze how attention and reasoning differ in low-resource language tasks.
- **Comparative Reasoning:** Compare linguistic reasoning to mathematical reasoning in LLMs.
- **Open Science:** All code, data, and results are open for reproducibility and further research.

## Future Development
The following items are planned for future development:

- **Dataset DOI:** Obtain a Digital Object Identifier (DOI) through a service like Zenodo to enable permanent, standardized citation of the synthetic Stoney Nakoda dataset.
- **Community Validation Interface:** Develop a web interface for native speakers to validate and provide feedback on model outputs.
- **RLHF Pipeline Enhancement:** Implement and refine the reward model components outlined in the mechanistic interpretability goals.
- **Human Preference Collection:** Set up a system for collecting human preferences to further align the model with native speaker expectations.
- **Multi-Modal Extensions:** Explore connections between language, images, and audio in the Stoney Nakoda context.

## Citation
If you use this project or dataset, please cite the Hugging Face dataset and this repository.

## License
MIT License (see LICENSE file)

## Contact
For questions or collaboration, open an issue or contact [HarleyCooper](https://huggingface.co/HarleyCooper). 