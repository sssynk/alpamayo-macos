# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device utilities for cross-platform support (CUDA, MPS, CPU)."""

import os
import torch


def get_default_device() -> torch.device:
    """Get the best available device for the current platform.

    Returns:
        torch.device: The best available device (cuda > mps > cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_type(device: torch.device | str | None = None) -> str:
    """Get the device type string for autocast.

    Note: MPS has limited autocast support, so we return "cpu" for MPS
    to disable autocast (which is the safest option for numerical stability).

    Args:
        device: The device to get the type for. If None, uses default device.

    Returns:
        str: Device type string suitable for torch.amp.autocast.
    """
    if device is None:
        device = get_default_device()

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        return "cuda"
    else:
        # For MPS and CPU, use "cpu" device type for autocast
        # MPS autocast is limited and can cause numerical issues
        return "cpu"


def get_default_dtype(device: torch.device | str | None = None) -> torch.dtype:
    """Get the recommended dtype for the device.

    Args:
        device: The device to get the dtype for. If None, uses default device.

    Returns:
        torch.dtype: Recommended dtype for the device.
    """
    if device is None:
        device = get_default_device()

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        return torch.bfloat16
    elif device.type == "mps":
        # MPS has better float16 support than bfloat16
        return torch.float16
    else:
        return torch.float32


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all devices.

    Args:
        seed: The random seed to use.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS doesn't have a separate manual_seed function; torch.manual_seed covers it


def setup_mps_fallback() -> None:
    """Enable MPS fallback for unsupported operations.

    This should be called before running inference on MPS to ensure
    unsupported operations fall back to CPU gracefully.
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def autocast_context(device: torch.device | str | None = None, enabled: bool = True):
    """Get an appropriate autocast context for the device.

    Args:
        device: The device to create context for. If None, uses default device.
        enabled: Whether to enable autocast.

    Returns:
        torch.autocast context manager.
    """
    device_type = get_device_type(device)
    dtype = get_default_dtype(device)

    # For non-CUDA devices, disable autocast for numerical stability
    if device_type != "cuda":
        enabled = False

    return torch.autocast(device_type, dtype=dtype, enabled=enabled)
