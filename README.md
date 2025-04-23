# neuromodAI
## 🧑‍🎓 Author

Riccardo Casciotti  
Master’s Thesis, Politecnico di Milano  
Advisors: Prof. Alberto Antonietti, Prof. Alessandra Pedrocchi, Francesco De Santis

## 📅 Academic Year

2024–2025

# 🧠 Brain-Inspired Continual Learning in Hebbian Deep Neural Networks

A biologically inspired approach to overcome catastrophic forgetting in **Hebbian-based deep neural networks**, developed as part of Riccardo Casciotti’s Master's thesis in Computer Science and Engineering at Politecnico di Milano (2024–2025).

## 📄 Overview

Modern Artificial Neural Networks (ANNs) still struggle with **catastrophic forgetting** — the tendency to forget previously learned tasks when learning new ones. Inspired by **Hebbian learning** and **neuromodulation** mechanisms found in the human brain, this project implements novel mechanisms within a biologically plausible architecture, **SoftHebb**, to mitigate forgetting in a **task-free continual learning** setup.

## 🚀 Objectives

- Implement and test **neuromodulation-inspired plasticity** control mechanisms.
- Apply a **multi-head architecture** to isolate task-specific learning at the classifier level.
- Validate the model's performance on **incremental image classification tasks** using CIFAR-10 and CIFAR-100 datasets.

## 🧪 Methods

### 🧬 SoftHebb Architecture
- A deep convolutional neural network trained in an **unsupervised** manner based on correlations in the input.
- Mimics **Hebbian synaptic plasticity** rules.

### 🔁 Kernel Plasticity Neuromodulation
A dopamine-inspired approach for selective weight update:
1. **Track kernel weight changes** over training intervals.
2. **Rank kernels** by their cumulative activation values.
3. **Modulate learning** by:
   - Reducing plasticity of important kernels.
   - Increasing plasticity for less relevant ones.

### 🧠 Multi-Head Architecture
- Each task has its own **head** (final classifier layer).
- During inference, the model automatically selects the most appropriate head using an unsupervised scoring mechanism.

## 📊 Experimental Setup

- Benchmarked on **CIFAR-10** and **CIFAR-100**.
- Task-based incremental learning:
  - Varying number of tasks (2 to 10).
  - Varying network depths (3 to 11 layers).
  - Varying number of classes per task.

### Models Compared:
- `V-model`: Vanilla SoftHebb (no continual learning support)
- `M-model`: Multi-head only
- `KP-model`: Kernel plasticity neuromodulation only
- `KPM-model`: Combines multi-head and neuromodulation

## 📈 Key Results

- The **KPM-model** outperforms all other variants in retaining earlier task performance.
- It **balances memory retention and adaptability**, addressing catastrophic forgetting effectively.
- Gains were most significant with 6-layer networks and moderate task complexity.
- Models converge in **one unsupervised epoch**, offering **fast training** and **efficient learning**.

## 📚 References

1. Adrien Journé et al. _Hebbian Deep Learning Without Feedback_.
2. Alex Krizhevsky. _Learning Multiple Layers of Features from Tiny Images_.
3. Arjun Magotra, Juntae Kim. _Neuromodulated Dopamine Plastic Networks..._
4. Liyuan Wang et al. _A Comprehensive Survey of Continual Learning_.


# 📦 Setup Instructions
## 📁 Project Structure

```
neuromodAI-main/
├── SoftHebb-main/           # Core Hebbian model and engine code
│   ├── environment_pytorch==1.7.1.yml
│   ├── train.py             # Main training pipeline
│   └── model.py             # SoftHebb model definition
├── batches/                 # Experiment automation scripts and testing
│   ├── testing.py           # Evaluation scripts for experiments
│   ├── t_hyper.py           # Hyperparameter tuning or task-specific setup
│   ├── stats.py             # Statistical analysis of experimental results
│   └── latex_tables.py      # LaTeX table generation for paper-ready output
├── softhebb_env/            # Conda environment files
│   ├── conda_reqs.txt
│   └── pip_reqs.txt
```

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/neuromodAI.git
cd neuromodAI/neuromodAI-main
```

### 2. Create and Activate the Conda Environment

```bash
conda create --name softhebb_env python=3.8
conda activate softhebb_env
pip install -r softhebb_env/pip_reqs.txt
```


## 📊 Model Testing & Configuration (via `t_hyper.py`)

The script `t_hyper.py` in the `batches/` folder is used to configure and manage continual learning experiments, particularly for hyperparameter tuning or automated experiment runs across different datasets and task configurations.

### 🛠️ Configuration Parameters

These are the main parameters used in `t_hyper.py`:

| Parameter            | Description |
|---------------------|-------------|
| `classes_per_task`  | Number of classes associated with each task (e.g., 2, 4, 6). |
| `n_experiments`     | Number of repeated runs per configuration. Default: 80. |
| `n_tasks`           | Total number of tasks to be learned incrementally. |
| `evaluated_tasks`   | List of task indices to evaluate performance on. |
| `data_num`          | Use `1` for single dataset, `2` for multi-dataset continual learning. |
| `dataset` / `dataset2` | Dataset identifiers (e.g., "C100", "C10", "STL10"). |
| `training_mode`     | Strategy for learning tasks (e.g., 'consecutive'). |
| `top_k`             | Fraction of top kernels to protect from overwriting. |
| `topk_lock`         | Boolean to freeze top-k kernel weights. |
| `high_lr` / `low_lr`| Learning rate modifiers for plastic vs important kernels. |
| `t_criteria`        | Importance criterion: 'activations' or 'KSE'. |
| `delta_w_interval`  | Interval (in batches) for tracking kernel updates. |

Modify these parameters directly in `t_hyper.py` to tailor your experimental design.

## 📊 Results

- Validated on CIFAR-10 and CIFAR-100 datasets.
- Models tested for performance across varying tasks, layers, and class-per-task settings.
- Outputs and plots saved in `SoftHebb-main/Tables/` and `SoftHebb-main/ppgraphs/`.

## 📊 Post-Experiment Analysis

After experiments are completed, you can analyze and summarize the results using the following utilities:

### `stats.py`
This script computes and logs statistics on experimental results, such as average accuracy, p-values, and confidence intervals. It is particularly useful for comparing models across tasks.

**Usage:**
```bash
cd batches
python stats.py
```

### `latex_tables.py`
This script generates LaTeX-formatted tables suitable for inclusion in academic papers. It reads performance metrics and outputs tables summarizing the results.

**Usage:**
```bash
cd batches
python latex_tables.py
```
You can then include the generated `.tex` file in your LaTeX documents for clean table presentation.





---
