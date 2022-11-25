#!/bin/bash

#$ -l rt_G.large=1
#$ -l h_rt=72:00:00 
#$ -j y
#$ -cwd
# ジョブ名
#$ -N job_name
#$ -m a
#$ -m b
#$ -m e
#$ -v GPU_COMPUTE_MODE=1

# 標準出力先
#$-o logs/stdout.txt

# qrsh -g gcc50441 -l rt_G.large=1 -l h_rt=4:00:00 

source /etc/profile.d/modules.sh
module load cuda/11.6/11.6.2
module load cudnn/8.3/8.3.3
module load gcc/11.2.0
module load python/3.8/3.8.13
module load nccl/2.12/2.12.10-1
module load openmpi/4.0.5

echo "python version is"
python3 -V

# pip3 install -r requirements.txt
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

cd /home/acd14210nv/baseline

echo "working directory is {$PWD}"
ALPHA=0
MODEL=conv
python src/main.py --alpha $ALPHA --model $MODEL --mode train
# python src/main.py --alpha $ALPHA --model $MODEL --mode eval
