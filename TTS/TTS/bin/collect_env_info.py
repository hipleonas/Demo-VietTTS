"""Các hàm thu thập thông số của môi trường + Hệ thống"""

import os
import platform
import sys
from datetime import datetime

import json
import numpy as np
import torch

import TTS


def cuda_info():
    return {
        "VERSION : ": torch.version.cuda,
        "CUDNN : ": torch.backends.cudnn.version(),
        "AVAILABLE : ": torch.cuda.is_available(),
        "GPU : ": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    }
def system_info():
    return {
        "OS": platform.system(),
        "ARCHITECTURE": platform.architecture(),
        "VERSION": platform.version(),
        "PROCESSOR": platform.processor(),
        "PYTHON": platform.python_version(),
    }
def package_info():
    return {
        "Numpy": np.__version__,
        "PyTorch_version": torch.__version__,
        "PyTorch_debug": torch.version.debug,
        "TTS": TTS.__version__,
    }
def main():
    system_details = {
        "CUDA : ": cuda_info(),
        "SYSTEM :": system_info,
        "PACKAGES : ": package_info(),
    }
    print(json.dumps(system_details, indent=4, sort_keys= True))

if __name__ == "__main__":  
    main()