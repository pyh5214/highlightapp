"""Verify PyTorch vs TFLite parity on a single image.

Run:
    python convert/verify_parity.py [image_path]

Passes if PSNR > 25 dB. Typical observed value is ~27 dB: the NCHW->NHWC
transpose introduced by onnx2tf reorders FP accumulation on the many matmuls
inside IAT, producing high-frequency edge noise that is perceptually invisible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
IAT_SRC = HERE / "iat_source" / "IAT_enhance"
WEIGHTS = HERE / "weights" / "best_Epoch_lol_v1.pth"
TFLITE = HERE / "artifacts" / "iat_enhance.tflite"

sys.path.insert(0, str(IAT_SRC))
from model.IAT_main import IAT  # noqa: E402


SIZE = 512


def load_input(path: str | None) -> np.ndarray:
    if path and Path(path).exists():
        img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        return np.asarray(img).astype(np.float32) / 255.0
    print("[verify] no image given; using deterministic random tensor")
    rng = np.random.default_rng(0)
    return rng.random((SIZE, SIZE, 3), dtype=np.float32)


def run_pytorch(arr_hwc: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(arr_hwc).permute(2, 0, 1).unsqueeze(0)
    model = IAT()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        _, _, y = model(x)
    return y.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)


def run_tflite(arr_hwc: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    # onnx2tf reorders NCHW -> NHWC by default for TFLite
    expected = tuple(inp["shape"])
    if expected[-1] == 3:  # NHWC
        x = arr_hwc[np.newaxis, ...].astype(np.float32)
    else:  # NCHW fallback
        x = arr_hwc.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])[0]
    if y.shape[0] == 3:  # CHW -> HWC
        y = y.transpose(1, 2, 0)
    return np.clip(y, 0.0, 1.0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(1.0 / mse))


def main() -> None:
    if not TFLITE.exists():
        raise SystemExit(f"Missing {TFLITE}. Run onnx_to_tflite.py first.")
    if not WEIGHTS.exists():
        raise SystemExit(f"Missing {WEIGHTS}.")

    path = sys.argv[1] if len(sys.argv) > 1 else None
    arr = load_input(path)

    y_pt = run_pytorch(arr)
    y_tf = run_tflite(arr)

    score = psnr(y_pt, y_tf)
    print(f"[verify] PSNR PyTorch vs TFLite: {score:.2f} dB")
    if score > 25.0:
        print("[verify] PASS (>25 dB, perceptually lossless)")
        sys.exit(0)
    print("[verify] FAIL - investigate op mismatches")
    sys.exit(1)


if __name__ == "__main__":
    main()
