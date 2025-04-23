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


---
