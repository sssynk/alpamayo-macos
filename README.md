<div align="center">

# Alpamayo 1 - macOS Fork

### Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving

[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Alpamayo--R1--10B-blue)](https://huggingface.co/nvidia/Alpamayo-R1-10B)
[![arXiv](https://img.shields.io/badge/arXiv-2511.00088-b31b1b.svg)](https://arxiv.org/abs/2511.00088)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

**This is an unofficial macOS-compatible fork of [NVIDIA's Alpamayo](https://github.com/NVlabs/alpamayo)**

</div>

## macOS Compatibility

This fork modifies the original Alpamayo codebase to run on Apple Silicon Macs using Metal Performance Shaders (MPS). The key changes include:

- Replaced CUDA-specific autocast decorators with device-agnostic alternatives
- Added automatic device detection (CUDA > MPS > CPU)
- Made Flash Attention optional (CUDA-only dependency)
- Relaxed Python version requirements to support 3.12+

### Requirements for macOS

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.12+ |
| **Mac** | Apple Silicon (M1/M2/M3) with 32GB+ unified memory (64GB recommended) |
| **macOS** | macOS 12.3+ (for MPS support) |

> **Note**: This fork has been tested on M2 Max with 64GB RAM. Macs with less memory may encounter out-of-memory errors.

## Installation on macOS

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/alpamayo-macos.git
cd alpamayo-macos
```

### 3. Set up the environment

```bash
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active
```

### 4. Authenticate with HuggingFace

The model requires access to gated resources. Request access here:
- [Physical AI AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)

Then authenticate:

```bash
pip install huggingface_hub
huggingface-cli login
```

## Running Inference on macOS

```bash
python src/alpamayo_r1/test_inference.py
```

The script will automatically:
- Detect your device (MPS on Apple Silicon)
- Use appropriate dtype (float16 for MPS)
- Enable MPS fallback for unsupported operations

### Expected output

```
Using device: mps, dtype: torch.float16
Loading dataset for clip_id: 030c760c-ae38-49aa-9ad8-f5650a545d26...
Dataset loaded.
...
Chain-of-Causation (per trajectory):
[['Nudge to the left to pass the stopped truck encroaching into the lane.']]
minADE: X.XXX meters
```

## Changes from Original Repository

| File | Changes |
|------|---------|
| `device_utils.py` | **NEW** - Device detection and cross-platform utilities |
| `test_inference.py` | Uses device_utils for automatic device/dtype selection |
| `action_space/unicycle_accel_curvature.py` | Replaced CUDA autocast with CPU (disabled) |
| `action_space/utils.py` | Replaced CUDA autocast with CPU (disabled) |
| `pyproject.toml` | Made flash-attn optional, relaxed version constraints |

## Troubleshooting

### MPS Fallback Warnings

You may see warnings like:
```
The operator 'aten::XXX' is not currently implemented for the MPS device.
```

This is expected - the code automatically falls back to CPU for unsupported operations via `PYTORCH_ENABLE_MPS_FALLBACK=1`.

### Out of Memory

If you run out of memory:
1. Reduce `num_traj_samples` in test_inference.py
2. Close other applications
3. Consider using a Mac with more unified memory

### Numerical Differences

Results may differ slightly from CUDA due to:
- Different precision handling (float16 vs bfloat16)
- MPS-specific operation implementations
- Fallback to CPU for some operations

---

## Original README Content

_The following is preserved from the original NVIDIA repository:_

---

> **Please read the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) first!**
> The model card contains comprehensive details on model architecture, inputs/outputs, licensing, and tested hardware configurations.

## Requirements (Original/CUDA)

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.12.x |
| **GPU** | NVIDIA GPU with 24 GB VRAM (e.g., RTX 3090, RTX 4090, A5000, H100) |
| **OS** | Linux (tested) |

## License

- **Inference code**: Apache License 2.0 - see [LICENSE](./LICENSE) for details.
- **Model weights**: Non-commercial license - see [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) for details.

## Citation

If you use Alpamayo 1 in your research, please cite:

```bibtex
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and others},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```
