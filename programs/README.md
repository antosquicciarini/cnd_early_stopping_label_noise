# Memorization Under Label Noise

This repository contains code and experiments for studying **early stopping** and related performance metrics in deep learning under **label noise** conditions.  
It includes experiment scripts, loss function definitions, graph-based entropy analysis, and SLURM job configurations for large-scale experiments on GPUs.

---

## 📂 Project Structure

- **`early_stopping_preformance.py`**  
  Generates the final result tables across datasets and experiments.

- **Experiment Folders**  
  - **NEWS:** `NEWS_CND_yu19_main_dropout_final`  
  - **CIFAR10:** `early_stopping_variab_no_early_stopping_preact_fin_no_mil`  
  - **CIFAR100:** `early_stopping_variab_no_early_stopping_preact_fin_no_mil`  
  - **TinyImagenet:** `TinyImagenet_CND`  

- **`cloud_TOML_basic.sh`**  
  SLURM job script for HPC cluster execution.  
  Handles:
  - GPU allocation (A100)
  - Virtual environment creation
  - PyTorch & dependency installation
  - Job execution with dataset-specific settings

- **`graph_entropy.py`**  
  Implements entropy-based graph metrics (Von Neumann & Shannon entropy) for analyzing DNN behavior.

- **`loss_functions.py`**  
  Customizable loss functions:
  - CrossEntropy
  - Mean Absolute Error (MAE) with logits

- **`loss_sesivity.py`**  
  Computes gradient-based **loss sensitivity** across batches.

- **`parameter_settings.py`**  
  Utility for managing experiment parameters, logging, and reproducibility.

- **`performances_plot_generation.py`**  
  Script to reload saved training results (`performances.pkl`, `args.json`) and generate diagnostic plots.

- **`prediction_changes.py`**  
  Tracks prediction changes between epochs for stability analysis.

- **`logs/`**  
  Contains raw simulation logs from experiments.

- **`LICENSE`**  
  MIT License.

---

## ⚙️ Parameters for Final Simulations

- **NEWS Dataset**  
  - Patience: `100`  
  - Window: `4`  
  - CND Percentile: `10%`

---

## 🚀 Usage

### Run Experiments (HPC with SLURM)
Submit a job:

```bash
sbatch cloud_TOML_basic.sh