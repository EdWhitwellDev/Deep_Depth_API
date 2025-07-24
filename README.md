# 👁️‍🗨️ OptoAPI: Depth & 3D Tools for Optometry

**OptoAPI** is a lightweight web API designed to support core optometry applications such as:
- **Depth estimation**
- **Stereo trajectory visualization**
- **3D scene reconstruction**

Built for educational, research, and prototyping purposes in clinical and applied vision sciences.

---

## 🚀 Built on NVIDIA FoundationStereo

> ⚠️ **This project is built on top of [NVIDIA's FoundationStereo](https://github.com/NVlabs/foundation-stereo), a powerful stereo depth estimation model designed for high-fidelity 3D perception.**

We leverage FoundationStereo for its robust and generalizable depth prediction across diverse stereo inputs. All core depth functionality in this API is a wrapper around NVIDIA’s released model.

---

## 🧠 Core Features

- 🔍 **Stereo Depth Estimation** using NVIDIA's FoundationStereo
- 🧭 **Trajectory Tools** for analyzing eye movement or camera motion in 3D
- 🧱 **3D Reconstruction** from stereo pairs or sequences
- 🌐 **FastAPI**-based RESTful interface for easy integration with front-ends or clinical tools



Some weird torch nvidia compat stuff just run this or equivalent for you nvidia architecture once the env is active
RTX 5070
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RTX 4060
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
