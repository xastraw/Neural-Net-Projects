#!/bin/bash
#SBATCH --job-name=xander-tim-test1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=xandertimtest1.out



source /home/jhub/jhub-venv/bin/activate
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate neural-net-minst


export LD_LIBRARY_PATH=/home/jhub/jhub-venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/jhub/jhub-venv/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/jhub/jhub-venv/lib/python3.10/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH



jupyter execute train.ipynb