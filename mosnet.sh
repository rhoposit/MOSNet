#!/bin/sh
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --time=4-00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/
export CUDNN_HOME=/opt/cuDNN-7.1_9.1/
#export CUDA_HOME=/opt/cuda-8.0.44
#export CUDNN_HOME=/opt/cuDNN-7.1_9.1/
export STUDENT_ID=$(whoami)
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTHON_PATH=$PATH

#python apply_model.py $1
#python utils.py
python get_scores.py

#python train_rep.py --model CNN --data LA --feats $1
#python train.py --model $1 --data LA --feats orig --batch_size $2 --alpha $3

