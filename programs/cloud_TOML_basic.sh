#!/bin/bash
#SBATCH --job-name=13_nov_KDE
#SBATCH --output=outputs/training_output_13_nov_KDE.txt
#SBATCH --error=outputs/training_error_13_nov_KDE.txt
#SBATCH --ntasks=1
#SBATCH --partition=standard-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.squicciarini@alumnos.upm.es

#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Load necessary modules if not already loaded
module --force purge
module load apps/2021
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
#module load TensorFlow/2.8.4-foss-2021b-CUDA-11.4.1

# Create and activate a Python virtual environment
VENV_PATH="venv"  

# Check if the virtual environment exists; if not, create it
if [ ! -d "$VENV_PATH" ]; then
  python -m venv $VENV_PATH
fi

# Activate the virtual environment
source $VENV_PATH/bin/activate


# Install the packages you need in the virtual environment
pip install --upgrade cuda-python
#pip uninstall torch # ==2.2.1 # 2.1.0
#pip install torch==1.7.1 torchvision  # This should install a compatible version with torch 2.1.0
#pip install torch==2.3.0 torchvision==0.18.0
#pip install torch==2.2.2 torchvision==0.17.20 torchaudio==2.2.2
#pip install torch==2.0.1 torchvision==0.15.2a0 torchaudio==2.0.2
#pip install --upgrade torch torchvision
#pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
#pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html


pip install seaborn
pip install networkx
pip install scikit-learn
pip install --upgrade seaborn numexpr bottleneck
pip install numpy==1.26.4
#pip install numpy==1.24.0
#pip install --upgrade pybind11


#pip install --upgrade --no-deps -r requirements.txt

SCRIPT_PATH="main.py"
#JOB_NAMES=("CIFAR10_none" "CIFAR10_label_noise" "CIFAR10_hard_noise" "FashionMNIST_none" "FashionMNIST_label_noise" "FashionMNIST_hard_noise" "CIFAR100_none" "CIFAR100_label_noise" "CIFAR100_hard_noise")
#JOB_NAMES=("FashionMNIST_CND_mean" "FashionMNIST_CND_PMF")
#JOB_NAMES=("webvision_debug")
#JOB_NAMES=("webvision")
#JOB_NAMES=("webvision_debug")
#JOB_NAMES=("CIFAR100N_CND_PMF")
JOB_NAMES=("basic_dataset_KDE")



# Loop over each job name and run the Python script
for JOB_NAME in "${JOB_NAMES[@]}"
do
  echo "Running job: $JOB_NAME"
  TF_GPU_ALLOCATOR=cuda_malloc_async python $SCRIPT_PATH --json-setting-cloud $JOB_NAME
done


# Deactivate the virtual environment if you activated one
deactivate

