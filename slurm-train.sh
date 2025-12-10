#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=test1%j.out



source /home/jhub/jhub-venv/bin/activate

export LD_LIBRARY_PATH=/home/jhub/jhub-venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/jhub/jhub-venv/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/jhub/jhub-venv/lib/python3.10/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH



jupyter execute test1.ipynb
