# NVIDIA Driver & Docker GPU Setup for Blackwell GPUs

# Ubuntu 22.04 LTS | RTX PRO 6000 / RTX 50-series

# ================================================

## Part 1: Install NVIDIA Driver (requires reboot)

```bash
# 1. Update & install essentials
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential dkms

# 2. Blacklist Nouveau
sudo bash -c "echo 'blacklist nouveau' >> /etc/modprobe.d/blacklist-nouveau.conf"
sudo bash -c "echo 'options nouveau modeset=0' >> /etc/modprobe.d/blacklist-nouveau.conf"
sudo update-initramfs -u

# 3. Add NVIDIA repo & install OPEN kernel modules (required for Blackwell)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y nvidia-driver-575 nvidia-dkms-575-open

# 4. Reboot
sudo reboot
```

---

## Part 2: Install NVIDIA Container Toolkit (run after reboot)

```bash
# 1. Verify driver is working
nvidia-smi

# 2. Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. Install the toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# 4. Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# 5. Restart Docker
sudo systemctl restart docker

# 6. Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Verification

After both parts complete, you should see your GPU in both:

- `nvidia-smi` (host)
- `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` (container)
