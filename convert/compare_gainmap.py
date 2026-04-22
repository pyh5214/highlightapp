"""Compare direct upscale vs gain-map upscale (sharpness test).

Saves 3-up rows: [original | direct IAT+WB (soft) | gain-map IAT+WB (sharp)].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
TFLITE = HERE / "artifacts" / "iat_enhance.tflite"
OUT_DIR = HERE / "artifacts" / "compare_gainmap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 512
MAX_GAIN = 20.0


def load_and_resize(path: Path, max_edge: int = 2048) -> Image.Image:
    """Keep source at closer to native resolution to demonstrate sharpness gain."""
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


def run_tflite(arr_hwc) -> np.ndarray:
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(TFLITE))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    x = arr_hwc[np.newaxis, ...].astype(np.float32)
    it.set_tensor(inp["index"], x)
    it.invoke()
    y = it.get_tensor(out["index"])[0]
    if y.shape[0] == 3:
        y = y.transpose(1, 2, 0)
    return np.clip(y, 0, 1)


def gray_world_wb(img):
    r, g, b = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
    gray = (r + g + b) / 3.0
    gains = np.array([gray / max(r, 1e-6), gray / max(g, 1e-6), gray / max(b, 1e-6)])
    return (img * gains).clip(0, 1).astype(np.float32)


def direct_upscale(enhanced_512, box, out_w, out_h) -> np.ndarray:
    """Current pipeline: crop content then bilinear upscale to original size."""
    px, py, cw, ch = box
    content = enhanced_512[py:py + ch, px:px + cw]
    im = Image.fromarray((content * 255).astype(np.uint8))
    return np.asarray(im.resize((out_w, out_h), Image.BILINEAR)).astype(np.float32) / 255.0


def gain_map_enhance(orig_full_arr, enhanced_low_arr) -> np.ndarray:
    """Bilinear-upscale the (enhanced/original) ratio and apply to full-res original."""
    H, W = orig_full_arr.shape[:2]
    ch, cw = enhanced_low_arr.shape[:2]

    orig_low = np.asarray(
        Image.fromarray((orig_full_arr * 255).astype(np.uint8)).resize((cw, ch), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    enh_full_blur = np.asarray(
        Image.fromarray((enhanced_low_arr * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    orig_full_blur = np.asarray(
        Image.fromarray((orig_low * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    eps = 1.0 / 255.0
    gain = np.clip(enh_full_blur / np.maximum(orig_full_blur, eps), 0.0, MAX_GAIN)
    return np.clip(orig_full_arr * gain, 0, 1).astype(np.float32)


def label_strip(width, text, h=32):
    strip = Image.new("RGB", (width, h), (32, 32, 32))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), text, fill=(240, 240, 240), font=font)
    return strip


def process_one(path: Path) -> Image.Image:
    print(f"[gain] {path.name}")
    img = load_and_resize(path)
    orig_arr = np.asarray(img).astype(np.float32) / 255.0
    W, H = img.size

    boxed, box = letterbox(img)
    enhanced_512 = run_tflite(boxed)

    # Content crop (low-res enhanced matching what the model saw)
    px, py, cw, ch = box
    content_low = enhanced_512[py:py + ch, px:px + cw]
    content_low_wb = gray_world_wb(content_low)

    # Variant 1: direct bilinear upscale (current pipeline)
    direct = direct_upscale(enhanced_512, box, W, H)
    direct_wb = gray_world_wb(direct)

    # Variant 2: gain-map upscale
    gain = gain_map_enhance(orig_arr, content_low_wb)

    lbl_h = 32
    row = Image.new("RGB", (W * 3, H + lbl_h), (32, 32, 32))
    labels = ["original", "direct upscale + WB (soft)", "gain-map + WB (sharp)"]
    imgs = [orig_arr, direct_wb, gain]
    for i, (lbl, im) in enumerate(zip(labels, imgs)):
        row.paste(label_strip(W, lbl, lbl_h), (i * W, 0))
        row.paste(Image.fromarray((im * 255).astype(np.uint8)), (i * W, lbl_h))
    return row


def main():
    samples = HERE / "samples"
    sources = sorted(p for p in samples.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"[gain] {len(sources)} samples")
    for p in sources:
        row = process_one(p)
        out = OUT_DIR / f"{p.stem}.jpg"
        row.save(out, quality=92)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
