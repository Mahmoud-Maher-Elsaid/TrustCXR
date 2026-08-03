from __future__ import annotations

import sys

import pytest
import torch
import torchvision

from trustcxr import __version__


def test_supported_python_version() -> None:
    assert sys.version_info[:2] == (3, 12)


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_torch_and_torchvision_versions_are_available() -> None:
    assert torch.__version__
    assert torchvision.__version__


@pytest.mark.gpu
def test_cuda_gpu_is_available() -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1
    assert torch.cuda.get_device_properties(0).total_memory >= 7 * 1024**3
