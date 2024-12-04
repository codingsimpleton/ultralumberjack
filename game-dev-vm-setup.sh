#!/bin/bash
# game-dev-vm-setup.sh
# Skrypt konfiguracyjny dla maszyny wirtualnej Game Development

set -e

# Ustawienia logowania i bezpieczeństwa
log_file="/var/log/game-dev-vm-setup.log"
exec > >(tee -a $log_file) 2>&1

# Aktualizacja systemu
sudo apt-get update && sudo apt-get upgrade -y

# Instalacja niezbędnych zależności
sudo apt-get install -y \
    build-essential \
    git \
    wget \
    software-properties-common \
    cuda-toolkit \
    nvidia-container-toolkit \
    docker.io

# Konfiguracja NVIDIA GPU
sudo systemctl enable nvidia-container-runtime
sudo systemctl start nvidia-container-runtime

# Instalacja Lumberyard (symulowana ścieżka)
wget https://lumberyard-cdn.amazonaws.com/latest/lumberyard-installer.bin
chmod +x lumberyard-installer.bin
./lumberyard-installer.bin --silent --install-path=/opt/lumberyard

# Konfiguracja Vertex AI
pip install google-cloud-aiplatform vertexai

# Instalacja narzędzi deweloperskich
pip install torch tensorflow numpy scipy matplotlib

# Konfiguracja Docker dla GPU
sudo usermod -aG docker $USER
newgrp docker

# Ustawienie zmiennych środowiskowych
echo "export LUMBERYARD_HOME=/opt/lumberyard" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
source ~/.bashrc

# Benchmark GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Czyszczenie
sudo apt-get autoremove -y
sudo apt-get clean

echo "Konfiguracja maszyny wirtualnej Game Development zakończona sukcesem!"
