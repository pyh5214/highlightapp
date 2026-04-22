"""Compare two strategies on low-light samples:

  A+C: Adaptive intensity (brightness-based blend of original <- -> full IAT), no WB
  B:   Local-only IAT (no global branch / color matrix / gamma), 100% intensity

Saves per-sample 4-up rows: [original | full IAT | A+C | B].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
TFLITE_FULL = HERE / "artifacts" / "iat_enhance.tflite"
TFLITE_LOCAL = HERE / "artifacts" / "iat_enhance_local.tflite"
OUT_DIR = HERE / "artifacts" / "compare_ac_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 512


def load_and_resize(path: Path, max_edge: int = 1024) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        s = max_edge / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BILINEAR)
    return img


def letterbox(img: Image.Image):
    w, h = img.size
    scale = min(SIZE / w, SIZE / h)
    cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
    px, py = (SIZE - cw) // 2, (SIZE - ch) // 2
    canvas = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    canvas.paste(img.resize((cw, ch), Image.BILINEAR), (px, py))
    arr = np.asarray(canvas).astype(np.float32) / 255.0
    return arr, (px, py, cw, ch)


def unletterbox(processed_hwc: np.ndarray, box, out_w: int, out_h: int) -> np.ndarray:
    px, py, cw, ch = box
    cropped = processed_hwc[py:py + ch, px:px + cw]
    im = Image.fromarray((cropped.clip(0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((out_w, out_h), Image.BILINEAR)).astype(np.float32) / 255.0


def run_tflite(model_path: Path, arr_hwc: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(model_path))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    x = arr_hwc[np.newaxis, ...].astype(np.float32)
    it.set_tensor(inp["index"], x)
    it.invoke()
    y = it.get_tensor(out["index"])[0]
    if y.shape[0] == 3:
        y = y.transpose(1, 2, 0)
    return np.clip(y, 0, 1)


def adaptive_intensity(orig_arr: np.ndarray) -> float:
    b = float(orig_arr.mean())
    if b < 0.30:
        return 0.90
    if b < 0.50:
        return 0.50
    return 0.10


def label_strip(width: int, text: str, h: int = 28) -> Image.Image:
    strip = Image.new("RGB", (width, h), (32, 32, 32))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 5), text, fill=(240, 240, 240), font=font)
    return strip


def process_one(path: Path) -> Image.Image:
    print(f"[cmp] {path.name}")
    img = load_and_resize(path)
    orig_arr = np.asarray(img).astype(np.float32) / 255.0
    W, H = img.size

    boxed, box = letterbox(img)

    # Full IAT
    full_512 = run_tflite(TFLITE_FULL, boxed)
    full = unletterbox(full_512, box, W, H)

    # A+C: adaptive intensity blend, no WB
    alpha = adaptive_intensity(orig_arr)
    ac = (alpha * full + (1 - alpha) * orig_arr).clip(0, 1)

    # B: local-only at 100%
    local_512 = run_tflite(TFLITE_LOCAL, boxed)
    local = unletterbox(local_512, box, W, H)

    lbl_h = 28
    row = Image.new("RGB", (W * 4, H + lbl_h), (32, 32, 32))
    row.paste(label_strip(W, "original"), (0, 0))
    row.paste(label_strip(W, "Full IAT (100%)"), (W, 0))
    row.paste(label_strip(W, f"A+C adaptive (alpha={alpha:.2f})"), (W * 2, 0))
    row.paste(label_strip(W, "B local-only (100%)"), (W * 3, 0))
    row.paste(Image.fromarray((orig_arr * 255).astype(np.uint8)), (0, lbl_h))
    row.paste(Image.fromarray((full * 255).astype(np.uint8)), (W, lbl_h))
    row.paste(Image.fromarray((ac * 255).astype(np.uint8)), (W * 2, lbl_h))
    row.paste(Image.fromarray((local * 255).astype(np.uint8)), (W * 3, lbl_h))
    return row


def main():
    sources = []
    demo = HERE / "iat_source" / "IAT_enhance" / "demo_imgs"
    sources += sorted(demo.glob("*"))
    samples = HERE / "samples"
    if samples.exists():
        sources += sorted(
            p for p in samples.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    print(f"[cmp] {len(sources)} samples")
    for p in sources:
        row = process_one(p)
        out = OUT_DIR / f"{p.stem}.jpg"
        row.save(out, quality=90)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
