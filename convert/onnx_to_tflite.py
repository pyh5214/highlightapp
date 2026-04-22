"""Convert iat_enhance.onnx -> iat_enhance_fp16.tflite via onnx2tf.

Run:
    python convert/onnx_to_tflite.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ONNX = HERE / "artifacts" / "iat_enhance.onnx"
WORK = HERE / "artifacts" / "tflite_work"
FINAL = HERE / "artifacts" / "iat_enhance.tflite"


def main() -> None:
    if not ONNX.exists():
        raise SystemExit(f"Missing {ONNX}. Run export_onnx.py first.")
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(ONNX),
        "-o", str(WORK),
        "-tb", "tf_converter",  # flatbuffer_direct backend mis-converts the global branch
    ]
    print("[tflite] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Ship FP32 I/O + FP32 weights. TFLite GPU delegate on Android auto-promotes
    # compute to FP16 at runtime, so there is no perf loss vs *float16.tflite*,
    # and the FP32 variant runs on every CPU backend (XNNPACK, NNAPI).
    candidates = sorted(WORK.glob("*float32*.tflite"))
    if not candidates:
        candidates = sorted(WORK.glob("*.tflite"))
    if not candidates:
        raise SystemExit(f"No .tflite produced in {WORK}")

    chosen = candidates[0]
    shutil.copy2(chosen, FINAL)
    size_kb = FINAL.stat().st_size / 1024
    print(f"[tflite] wrote {FINAL} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
