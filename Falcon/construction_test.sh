srun -c1 -N1 --pty bash           # land on a node of the same partition
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

python - <<'PY'
import torch, platform, subprocess, os
print("torch imported OK on", platform.processor())
subprocess.run(["lscpu","--grep","AVX"], check=False)
PY
