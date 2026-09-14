# MUDAPIBench: Machine Unlearning for Deprecated API Knowledge in Large Language Models

Official code release for our paper 📄:

**[What Was Once Learned May Need to Be Unlearned: Machine Unlearning for Deprecated API Knowledge in Large Language Models]**

This repository provides **MUDAPIBench**, a benchmark for evaluating machine unlearning methods on deprecated API knowledge in large language models (LLMs), together with implementations of multiple machine unlearning methods.

MUDAPIBench contains more than 7,000 model-specific instances covering 145 verified deprecated-to-up-to-date API mappings across 8 widely used Python libraries. The benchmark evaluates whether an LLM can suppress deprecated API knowledge while preserving valid API knowledge, alternative code completions, and general coding capabilities.

## 🗂️ Project Structure

```text
.
|-- LICENSE
|-- README.md
|-- requirements.txt
|
`-- Machine_Unlearning/
    `-- Unlearning_LLM/
        `-- llm_unlearn/
            |-- run_all.sh                    # run all unlearning experiments
            |-- run_ds.sh                     # run experiments on DeepSeek-Coder
            |-- run_qwen.sh                   # run experiments on Qwen2.5-Coder
            |-- run_starcoder.sh              # run experiments on StarCoder2
            |-- run_unlearn_api_lora.py       # run API-specific LoRA unlearning
            |
            |-- methods/                      # implementations of unlearning methods
            |   |-- gradient_ascent.py        # Gradient Ascent (GA)
            |   |-- ascent_plus_descent.py    # Gradient Difference (GD)
            |   |-- ascent_plus_KLdivergence.py # KL-based unlearning (KL)
            |   |-- dpo.py                    # Direct Preference Optimization (DPO)
            |   |-- npo.py                    # Negative Preference Optimization (NPO)
            |   |-- prod.py                   # Preference-based Random Output Distillation (PROD)
            |   |-- simnpo.py                 # SimNPO
            |   |-- unlearning_argument.py    # unlearning arguments and configurations
            |   |-- utils.py                  # shared utilities
            |   `-- __init__.py
            |
            `-- utils/                         # shared utilities
```

## ⚙️ Environment Setup

### 1. Install Dependencies 🧩

Install the required dependencies from the provided requirements file:

```bash
conda create -n MUDAPIBench python=3.10 -y
conda activate MUDAPIBench
pip install -r requirements.txt
```

### 2. Model Availability 🤗

Please ensure that the corresponding Hugging Face models are available before running machine unlearning or evaluation.

| Model Key          | Hugging Face Model                     |
| ------------------ | -------------------------------------- |
| `deepseek-1.3b`    | `deepseek-ai/deepseek-coder-1.3b-base` |
| `starcoder2-3b`    | `bigcode/starcoder2-3b`                |
| `qwen2.5-coder-3b` | `Qwen/Qwen2.5-Coder-3B`                |

## 📦 MUDAPIBench

MUDAPIBench is designed specifically for studying the unlearning of deprecated API knowledge in code LLMs.

The benchmark contains:

- 7,000+ model-specific instances
- 145 verified deprecated-to-up-to-date API mappings
- 8 Python libraries
- Forgetting, generalization, and specificity evaluation scenarios

The benchmark covers the following libraries:

```text
PyTorch
TensorFlow
scikit-learn
SciPy
pandas
Seaborn
Transformers
NumPy
```

The benchmark data is included in:

```text
data/MUDAPIBench/<model_key>/
```

The benchmark construction pipeline starts from functions collected using Sourcegraph and applies automated filtering to retain instances where the target model actually exhibits the deprecated API behavior.

## 🚀 Quick Start

### Run Machine Unlearning

For example, to run all methods on three code LLMs:

```bash
cd MUDAPIBench/Machine_Unlearning/Unlearning_LLM/llm_unlearn

bash run_all.sh
```

Available unlearning methods:

```text
GA
GD
KL
RLFT
DPO
NPO
SimNPO
PROD
```

The methods cover several representative categories:

| Category                    | Methods          |
| --------------------------- | ---------------- |
| Gradient-based              | GA, GD           |
| Distribution Regularization | KL               |
| Random Supervision          | RLFT             |
| Preference Optimization     | DPO, NPO, SimNPO |
| Target Distribution         | PROD             |
