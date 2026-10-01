#!/usr/bin/env bash
# Установка в WSL2 (Ubuntu). Драйвер NVIDIA ставится в Windows, не в WSL.
set -e
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/stand-env
source ~/stand-env/bin/activate
pip install --upgrade pip
# RTX 50-й серии (Blackwell) нужна сборка PyTorch с CUDA 12.8 или новее
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate numpy
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
