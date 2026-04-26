"""
Claw Bot — Phase 4 Pre-flight Check
Verifies system readiness before downloading image models.

Checks:
  1. Python version
  2. Required libraries
  3. GPU presence and VRAM
  4. Free disk space
  5. ComfyUI installation
  6. Ollama (currently loaded model)
  7. Folder structure integrity
  8. Model registry validity
"""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def check(label: str, ok: bool, detail: str = ""):
    mark = "✅" if ok else "❌"
    print(f"  {mark} {label}" + (f" — {detail}" if detail else ""))
    return ok


def header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def section(title: str):
    print(f"\n--- {title} ---")


def main():
    header("CLAW BOT — PHASE 4 PRE-FLIGHT CHECK")
    print(f"Project root: {PROJECT_ROOT}")

    all_ok = True

    # ---- Python ----
    section("Python")
    py_version = sys.version_info
    ok = py_version.major == 3 and py_version.minor == 11
    check(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}",
          ok, "Expected 3.11.x" if not ok else "")
    all_ok &= ok

    # ---- Required libraries ----
    section("Required Python libraries")
    required = ["discord", "ollama", "dotenv", "apscheduler", "aiohttp"]
    for lib in required:
        try:
            __import__(lib if lib != "dotenv" else "dotenv")
            check(lib, True)
        except ImportError:
            check(lib, False, "Run: pip install " + lib)
            all_ok = False

    # ---- GPU check (nvidia-smi) ----
    section("GPU (NVIDIA)")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 4:
            name, total_mb, free_mb, driver = parts[0], int(parts[1]), int(parts[2]), parts[3]
            check(f"GPU: {name}", True)
            check(f"Total VRAM: {total_mb} MB", True)
            check(f"Free VRAM: {free_mb} MB", free_mb > 4000,
                  "Need 4+ GB free for SDXL Turbo" if free_mb <= 4000 else "")
            check(f"Driver: {driver}", True)
            if free_mb <= 4000:
                all_ok = False
                print("     ⚠ Suggestion: close other GPU apps (browser, ComfyUI, games) before generating.")
    except FileNotFoundError:
        check("nvidia-smi found", False, "No NVIDIA driver? You need NVIDIA for SDXL Turbo.")
        all_ok = False
    except Exception as e:
        check("nvidia-smi query", False, str(e))
        all_ok = False

    # ---- Disk space ----
    section("Disk space")
    drive = str(PROJECT_ROOT)[:3]   # e.g. "E:\"
    total, used, free = shutil.disk_usage(drive)
    free_gb = free // (1024 ** 3)
    check(f"Drive {drive} free: {free_gb} GB",
          free_gb >= 15,
          "Need 15+ GB for SDXL Turbo + LoRA + workflows buffer" if free_gb < 15 else "")
    if free_gb < 15:
        all_ok = False

    # ---- Folder structure ----
    section("Folder structure")
    required_folders = [
        "00_Tools",
        "01_ComfyUI/ComfyUI_windows_portable",
        "02_Agent/venv",
        "02_Agent/modules",
        "02_Agent/modules/image_backends",
        "03_Models/ollama",
        "03_Models/ComfyUI/checkpoints",
        "03_Models/ComfyUI/loras",
        "03_Models/character_refs",
        "04_Outputs/scripts",
        "04_Outputs/storyboards",
        "05_Config",
        "06_Logs",
    ]
    for folder in required_folders:
        exists = (PROJECT_ROOT / folder).exists()
        check(folder, exists, "Missing!" if not exists else "")
        if not exists:
            all_ok = False

    # ---- ComfyUI install ----
    section("ComfyUI installation")
    comfy_dir = PROJECT_ROOT / "01_ComfyUI" / "ComfyUI_windows_portable"
    checks = [
        ("run_nvidia_gpu.bat", comfy_dir / "run_nvidia_gpu.bat"),
        ("ComfyUI main folder", comfy_dir / "ComfyUI"),
        ("python_embeded folder", comfy_dir / "python_embeded"),
    ]
    for label, path in checks:
        exists = path.exists()
        check(label, exists, "Missing!" if not exists else "")
        if not exists:
            all_ok = False

    # ---- Ollama / LLM ----
    section("Ollama (local LLM)")
    ollama_exe = PROJECT_ROOT / "00_Tools" / "ollama" / "ollama.exe"
    check("ollama.exe", ollama_exe.exists(), "Missing!" if not ollama_exe.exists() else "")
    if ollama_exe.exists():
        try:
            r = subprocess.check_output(
                [str(ollama_exe), "list"], timeout=10,
                stderr=subprocess.DEVNULL,
            ).decode()
            has_llama = "llama3.1:8b-instruct-q4_K_M" in r
            check("Llama 3.1 8B model pulled", has_llama,
                  "Run: ollama pull llama3.1:8b-instruct-q4_K_M" if not has_llama else "")
            if not has_llama:
                all_ok = False
        except Exception as e:
            check("Ollama list query", False, str(e))

    # ---- Model registry ----
    section("Model registry")
    reg_path = PROJECT_ROOT / "05_Config" / "models.json"
    if reg_path.exists():
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            check("models.json valid JSON", True)
            image_active = reg.get("image_backend", {}).get("active")
            check(f"Active image backend: {image_active}",
                  image_active is not None,
                  "No active image backend set" if image_active is None else "")
        except Exception as e:
            check("models.json valid JSON", False, str(e))
            all_ok = False
    else:
        check("models.json exists", False, "Missing!")
        all_ok = False

    # ---- Summary ----
    header("SUMMARY")
    if all_ok:
        print("  🎉 All checks passed. System ready for Phase 4 downloads.\n")
        print("  Next step: 4.2 — Download SDXL Turbo checkpoint and LoRA.")
        return 0
    else:
        print("  ⚠ Some checks failed. Review the ❌ entries above before proceeding.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())