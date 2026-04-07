from __future__ import annotations

import os
import sys
from pathlib import Path

# ── NVIDIA CUDA DLL path fix (Windows + faster-whisper) ──────────────
# Must run BEFORE any import chain that loads faster-whisper / ctranslate2.
# The pip-installed nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels drop their
# DLLs in site-packages\nvidia\*\bin\, which Windows doesn't add to the DLL
# search path automatically. Without this, ctranslate2 fails with:
#   RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
if sys.platform == "win32":
    try:
        import nvidia.cublas
        import nvidia.cudnn
        _cublas_dir = os.path.join(os.path.dirname(nvidia.cublas.__file__), "bin")
        _cudnn_dir = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "bin")
        if os.path.isdir(_cublas_dir):
            os.add_dll_directory(_cublas_dir)
        if os.path.isdir(_cudnn_dir):
            os.add_dll_directory(_cudnn_dir)
        os.environ["PATH"] = (
            _cublas_dir + os.pathsep + _cudnn_dir + os.pathsep + os.environ.get("PATH", "")
        )
    except Exception as _e:
        print(f"[warn] Could not add NVIDIA DLL dirs: {_e}")

from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bootstrap MUST run before any module that touches app.config
from gui_app.bootstrap import setup_project_imports
setup_project_imports()

from gui_app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Shikshak AI Teacher")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())