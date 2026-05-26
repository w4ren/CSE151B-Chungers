# vLLM Setup

This pod can recreate the vLLM environment when needed. The environment below
is placed inside the repository so it is easy to remove and does not depend on a
global environment name.

Note for this pod: `nvidia-smi` reports GTX 1080 Ti GPUs. Current vLLM CUDA wheels require newer GPU compute capability than GTX 1080 Ti provides, so this environment can install the vLLM package but should not be expected to run accelerated vLLM inference on this hardware. Use the Transformers backend here unless the pod is moved to a newer GPU.

```bash
cd /home/mnt/BiomechAI/Wenhao/archive2/CSE151B-Chungers-wenhao-lfs

export PIP_CACHE_DIR="$PWD/.pip-cache"
python3 -m pip install --prefix ./.uv-local uv
export PATH="$PWD/.uv-local/bin:$PATH"
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"

uv venv .venv-vllm --python 3.11 --seed
source .venv-vllm/bin/activate
uv pip install --torch-backend=auto -r codex_added/requirements.txt
```

Verify:

```bash
python -c "import torch, vllm; print(vllm.__version__); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Smoke test:

```bash
python codex_added/scripts/run_inference.py \
  --config codex_added/configs/baseline.yaml \
  --backend vllm \
  --data data/public.jsonl \
  --output codex_added/results/vllm_smoke.jsonl \
  --limit 2
```
