#!/usr/bin/env bash
# env.sh  –  set up environment for Falcon
#srun -p tue.gpu.q --gpus=1 --pty bash 

module purge

# 1. Load the *exact* Python that matches the tool-chain
module load Python/3.11.3-GCCcore-12.3.0   # ← pin the ABI first
module load CUDA/12.1.1                    # runtime only
module load OpenCV/4.8.1-foss-2023a
module load BeautifulSoup/4.12.2-GCCcore-12.3.0 
module load numpy
# module load GDAL/3.7.1-foss-2023a

#tests for illegal construction
python - <<'PY'
import torch, platform, subprocess, shutil
print("Torch OK, GPU count:", torch.cuda.device_count(),
      "CPU flags:", platform.processor())
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "-L"], check=False, timeout=3)
PY

#subprocess.run(["nvidia-smi", "-L"])

# Virtual-env should contain only torchviz and matplotlib; everything else comes from the modules above.
venv_path="$HOME/Falcon/myvenv311"

if [ ! -d "$venv_path" ]; then            # first‑time setup
    python -m venv "$venv_path"
    source "$venv_path/bin/activate"

    # Tools
    pip install --upgrade pip wheel

    # PyTorch: CUDA 12.1 wheels
    pip install torch==2.1.2 \
                torchvision==0.16.2 \
                torchaudio==2.1.2 \
                --extra-index-url https://download.pytorch.org/whl/cu121

    # Your extra packages
    pip install torchviz==0.0.2 matplotlib==3.8.4
    pip install scikit-image
    pip install geopandas
    pip install pyarrow
    pip install seaborn
else
    source "$venv_path/bin/activate"
fi

export PYTHONPATH="$HOME/Falcon/FALcon-main:$PYTHONPATH"

