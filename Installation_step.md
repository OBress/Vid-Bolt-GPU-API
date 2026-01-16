# NVIDIA Driver & Docker GPU Setup for Blackwell GPUs

# Ubuntu 22.04 LTS | RTX PRO 6000 / RTX 50-series

# ===============================================

## Part 1: Install NVIDIA Driver, toolkit, & Docker

```bash
#!/bin/bash

# 1. Clean up existing installations to prevent conflicts
echo "--- Cleaning up old drivers and docker versions ---"
sudo apt-get purge -y '*nvidia*' 'libnvidia*' 'docker.io' 'docker-doc' 'docker-compose' 'docker-compose-v2' 'podman-docker' 'containerd' 'runc'
sudo apt autoremove -y

# 2. Install prerequisites
echo "--- Installing system prerequisites ---"
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential dkms curl ca-certificates gnupg linux-headers-$(uname -r)

# 3. Add NVIDIA Repository & Install Blackwell Driver
echo "--- Setting up NVIDIA Driver (575-Open) ---"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
# Blackwell REQUIRES the -open driver and utils for nvidia-smi
sudo apt install -y nvidia-driver-575-open nvidia-dkms-575-open nvidia-utils-575

# 4. Add Docker Official Repository & Install Docker
echo "--- Setting up Docker and Docker Compose ---"
sudo install -m 0.755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 5. Add NVIDIA Container Toolkit (GPU Docker Support)
echo "--- Setting up NVIDIA Container Toolkit ---"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit

# 6. Configure Docker to use NVIDIA Runtime
echo "--- Configuring Docker Runtime ---"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 7. Post-install Permissions & Performance Tweaks
echo "--- Finalizing permissions and persistence ---"
sudo usermod -aG docker $USER
sudo systemctl enable nvidia-persistenced
sudo systemctl start nvidia-persistenced

echo "--- SETUP COMPLETE ---"
echo "A reboot is required to load the Blackwell kernel modules."
echo "After rebooting, test with: nvidia-smi"
echo "Test Docker GPU with: docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi"

sudo reboot
```

---

## Verification

After both parts complete, you should see your GPU in both:

- `nvidia-smi` (host)
- `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` (container)

## Part 3: setup repos and env

git clone https://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@github.com/OBress/Vid-Bolt-GPU-API.git

cd Vid-Bolt-GPU-API

# shouldn't need this anymore cause models are in repo but if not

make setup-repos

cp .env.example .env
nano .env

## Part 4: build & run
git pull
docker system prune
docker compose build --no-cache
docker compose up -d
docker compose logs -f

# clean up old containers

docker system prune
