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

## Research Goals
- **Mechanistic Interp:** Visualize and analyze how attention and reasoning differ in low-resource language tasks.
- **Comparative Reasoning:** Compare linguistic reasoning to mathematical reasoning in LLMs.
- **Open Science:** All code, data, and results are open for reproducibility and further research.

## Citation
If you use this project or dataset, please cite the Hugging Face dataset and this repository.

## License
MIT License (see LICENSE file)

## Contact
For questions or collaboration, open an issue or contact [HarleyCooper](https://huggingface.co/HarleyCooper). 