# TrustCXR Final Windows Research Environment Reconstruction

This protocol reconstructs the validated `REPRODUCIBLE_RESEARCH_PIPELINE`. It does not claim
production, clinical, Linux, macOS, CPU-only, or alternative-CUDA equivalence.

Validated scope:

- Windows 11 with PowerShell 5.1 or PowerShell 7 orchestration
- CPython 3.12.10 x64
- NVIDIA CUDA-capable environment compatible with PyTorch CUDA 13.0 wheels
- repository-relative `.venv`

From a clean authorized repository checkout, run these commands in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip==26.2" "setuptools==78.1.0" "wheel==0.47.0"
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 "torch==2.12.1+cu130" "torchvision==0.27.1+cu130"
.\.venv\Scripts\python.exe -m pip install -r requirements\lock-final-research-windows-cu130.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\reproducibility\verify_final_environment.py
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts\project\validate_complete_stage_coverage.py
.\.venv\Scripts\python.exe -m pytest -q
```

The lock retains exact versions observed in the governed environment. Historical stage locks
remain immutable evidence; this file is the canonical final-release lock. The PyTorch index is
explicit because the `+cu130` builds are distributed through the official PyTorch CUDA index.

The verifier checks the lock hash, exact installed distributions, Python version, PyTorch CUDA
runtime, and the validated GPU model. It performs no model loading or inference. Frozen seeds,
splits, and configs support controlled reproduction, but bit-identical CUDA training is not
guaranteed.
