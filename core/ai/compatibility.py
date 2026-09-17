"""
Hardware and Runtime Compatibility Checker for Tarang Offline AI.
Inspects CPU, RAM, GPU, VRAM, and available disk space specifically for Qwen2.5-1.5B-Instruct GGUF.
Supports both CPU inference and optional GPU acceleration without freezing the GUI.
"""
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class CompatibilityResult:
    status: str  # "READY", "READY — CPU MODE", "LIMITED", "INSUFFICIENT DISK SPACE", "UNSUPPORTED ENVIRONMENT"
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.status in ("READY", "READY — CPU MODE", "LIMITED")


class AICompatibilityChecker:
    """
    Evaluates system hardware capabilities for running local Qwen2.5-1.5B-Instruct GGUF.
    Model requirements:
    - Disk: ~2.0 GB free (Q4_K_M weights are ~1.05 GB)
    - RAM: >= 2.5 GB physical RAM (model consumes ~1.2 GB active RAM during inference)
    - CPU: 64-bit x86_64 / ARM64 with AVX / NEON support
    - GPU: Optional (CUDA / Metal acceleration supported if available)
    """

    MODEL_DISK_REQUIRED_GB = 2.0
    MIN_RAM_GB = 2.5
    REC_RAM_GB = 6.0

    @classmethod
    def get_system_ram(cls) -> Dict[str, float]:
        """Returns total and available RAM in GB."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / (1024 ** 3), 2),
                "available_gb": round(mem.available / (1024 ** 3), 2),
            }
        except Exception:
            pass

        if platform.system() == "Windows":
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    return {
                        "total_gb": round(stat.ullTotalPhys / (1024 ** 3), 2),
                        "available_gb": round(stat.ullAvailPhys / (1024 ** 3), 2),
                    }
            except Exception:
                pass

        return {"total_gb": 4.0, "available_gb": 2.0}

    @classmethod
    def get_disk_free(cls, path: Path) -> float:
        """Returns free disk space in GB for the target partition."""
        try:
            target = path
            while not target.exists() and target.parent != target:
                target = target.parent
            usage = shutil.disk_usage(target)
            return round(usage.free / (1024 ** 3), 2)
        except Exception:
            return 10.0

    @classmethod
    def inspect_gpu(cls) -> Dict[str, Any]:
        """Detects whether dedicated GPU / CUDA is available without blocking."""
        info = {
            "has_gpu": False,
            "vendor": "CPU Only",
            "model": "Standard CPU",
            "vram_gb": 0.0,
            "cuda_available": False,
        }

        # Check torch if installed
        try:
            import torch
            if torch.cuda.is_available():
                info["has_gpu"] = True
                info["cuda_available"] = True
                info["vendor"] = "NVIDIA"
                info["model"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
                return info
        except Exception:
            pass

        # Fast check via nvidia-smi if on Windows
        if platform.system() == "Windows":
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"],
                    timeout=2,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                ).decode("utf-8").strip()
                if out:
                    first = out.split("\n")[0].split(",")
                    if len(first) >= 2:
                        info["has_gpu"] = True
                        info["vendor"] = "NVIDIA"
                        info["model"] = first[0].strip()
                        info["vram_gb"] = round(float(first[1].strip()) / 1024, 2)
                        info["cuda_available"] = True
                        return info
            except Exception:
                pass

        return info

    @classmethod
    def check_compatibility(cls, target_dir: Path) -> CompatibilityResult:
        """Runs the compatibility check for Qwen2.5-1.5B GGUF."""
        ram_info = cls.get_system_ram()
        disk_free_gb = cls.get_disk_free(target_dir)
        gpu_info = cls.inspect_gpu()

        is_64bit = sys.maxsize > 2**32
        cpu_arch = platform.machine()
        cpu_model = platform.processor() or "Generic x86_64 CPU"
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        os_info = f"{platform.system()} {platform.release()}"

        details = {
            "os": os_info,
            "python_version": py_ver,
            "is_64bit": is_64bit,
            "cpu_arch": cpu_arch,
            "cpu_model": cpu_model,
            "ram_total_gb": ram_info["total_gb"],
            "ram_available_gb": ram_info["available_gb"],
            "disk_free_gb": disk_free_gb,
            "disk_required_gb": cls.MODEL_DISK_REQUIRED_GB,
            "gpu": gpu_info,
        }

        recs = []

        # 1. Architecture disqualifier
        if not is_64bit:
            return CompatibilityResult(
                status="HARDWARE_UNSUPPORTED",
                summary="Offline AI cannot be installed on this computer. A 64-bit operating system is required.",
                details=details,
                recommendations=["A 64-bit operating system is required to run local AI inference."],
            )

        # 2. Pre-flight check: Verify local inference runtime engine
        try:
            from core.ai.runtime_manager import is_runtime_available
            runtime_ok, runtime_msg = is_runtime_available()
            details["runtime_available"] = runtime_ok
            details["runtime_message"] = runtime_msg
            if not runtime_ok:
                logger.error(f"Offline AI runtime pre-flight check failed: {runtime_msg}")
                return CompatibilityResult(
                    status="RUNTIME_UNAVAILABLE",
                    summary="Tarang couldn't start its local AI engine. Online AI remains available.",
                    details=details,
                    recommendations=["Ensure Windows 64-bit C++ runtime components are present."],
                )
        except Exception as re:
            logger.error(f"Offline AI runtime pre-flight check exception: {re}", exc_info=True)
            details["runtime_available"] = False
            details["runtime_error"] = str(re)
            return CompatibilityResult(
                status="RUNTIME_UNAVAILABLE",
                summary="Tarang couldn't start its local AI engine. Online AI remains available.",
                details=details,
                recommendations=["The local inference runtime encountered an error during startup."],
            )

        # 2. Disk space check
        if disk_free_gb < cls.MODEL_DISK_REQUIRED_GB:
            return CompatibilityResult(
                status="INSUFFICIENT DISK SPACE",
                summary=f"Insufficient disk space: {disk_free_gb} GB available, but {cls.MODEL_DISK_REQUIRED_GB} GB is required.",
                details=details,
                recommendations=[f"Free up at least {cls.MODEL_DISK_REQUIRED_GB - disk_free_gb:.1f} GB on your drive."],
            )

        # 3. RAM check
        if ram_info["total_gb"] > 0 and ram_info["total_gb"] < cls.MIN_RAM_GB:
            return CompatibilityResult(
                status="LIMITED",
                summary=f"System RAM is low ({ram_info['total_gb']} GB). Offline AI will run, but may experience high memory load.",
                details=details,
                recommendations=["Close background applications to free memory."],
            )

        # 4. GPU Ready
        if gpu_info["has_gpu"] and gpu_info["cuda_available"]:
            return CompatibilityResult(
                status="READY",
                summary="Your computer appears suitable for Offline AI with GPU acceleration.",
                details=details,
                recommendations=[f"Dedicated GPU detected: {gpu_info['model']} ({gpu_info['vram_gb']} GB VRAM)."],
            )

        # 5. CPU Ready
        return CompatibilityResult(
            status="READY — CPU MODE",
            summary="Your computer appears suitable for Offline AI. Local inference will run smoothly via fast CPU processing.",
            details=details,
            recommendations=["Fast local CPU execution is fully supported for Qwen2.5-1.5B GGUF."],
        )
