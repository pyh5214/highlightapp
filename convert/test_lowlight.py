"""Run IAT TFLite on diverse low-light samples and verify gray-world WB effect.

Produces a grid: [original | IAT raw | IAT + gray-world WB] per sample.
This mirrors what the Android app does at intensity=100%.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
TFLITE = HERE / "artifacts" / "iat_enhance.tflite"
OUT_DIR = HERE / "artifacts" / "lowlight_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 512


def load_and_resize(path: Path, max_edge: int = 1024) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        s = max_edge / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BILINEAR)
    return img


def letterbox(img: Image.Image) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    w, h = img.size
    scale = min(SIZE / w, SIZE / h)
    cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
    px, py = (SIZE - cw) // 2, (SIZE - ch) // 2
    canvas = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    canvas.paste(img.resize((cw, ch), Image.BILINEAR), (px, py))
    arr = np.asarray(canvas).astype(np.float32) / 255.0
    return arr, (px, py, cw, ch)


def unletterbox(processed_hwc: np.ndarray, box: tuple, out_w: int, out_h: int) -> np.ndarray:
    px, py, cw, ch = box
    cropped = processed_hwc[py:py + ch, px:px + cw]
    im = Image.fromarray((cropped.clip(0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((out_w, out_h), Image.BILINEAR)).astype(np.float32) / 255.0


def run_tflite(arr_hwc: np.ndarray) -> np.ndarray:
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


def gray_world_wb(img: np.ndarray) -> np.ndarray:
    """Same math as Kotlin ImagePipeline.grayWorldWhiteBalance."""
    r = img[..., 0].mean()
    g = img[..., 1].mean()
    b = img[..., 2].mean()
    gray = (r + g + b) / 3.0
    gains = np.array([gray / max(r, 1e-6), gray / max(g, 1e-6), gray / max(b, 1e-6)])
    out = (img * gains).clip(0, 1)
    return out.astype(np.float32)


def label_strip(width: int, text: str, height: int = 32) -> Image.Image:
    strip = Image.new("RGB", (width, height), (32, 32, 32))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), text, fill=(240, 240, 240), font=font)
    return strip


def process_one(path: Path) -> Image.Image:
    print(f"[test] {path.name}")
    img = load_and_resize(path)
    orig_arr = np.asarray(img).astype(np.float32) / 255.0

    boxed_arr, box = letterbox(img)
    enhanced_512 = run_tflite(boxed_arr)
    enhanced = unletterbox(enhanced_512, box, img.size[0], img.size[1])
    wb_corrected = gray_world_wb(enhanced)

    # Build 3-up row with labels
    w, h = img.size
    lbl_h = 28
    row = Image.new("RGB", (w * 3, h + lbl_h), (32, 32, 32))
    row.paste(label_strip(w, "original", lbl_h), (0, 0))
    row.paste(label_strip(w, "IAT (raw)", lbl_h), (w, 0))
    row.paste(label_strip(w, "IAT + gray-world WB", lbl_h), (w * 2, 0))
    row.paste(Image.fromarray((orig_arr * 255).astype(np.uint8)), (0, lbl_h))
    row.paste(Image.fromarray((enhanced * 255).astype(np.uint8)), (w, lbl_h))
    row.paste(Image.fromarray((wb_corrected * 255).astype(np.uint8)), (w * 2, lbl_h))
    return row


def main():
    sources: list[Path] = []
    demo_dir = HERE / "iat_source" / "IAT_enhance" / "demo_imgs"
    sources += sorted(demo_dir.glob("*"))
    samples_dir = HERE / "samples"
    if samples_dir.exists():
        sources += sorted(p for p in samples_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    print(f"[test] found {len(sources)} samples")
    rows = []
    for p in sources:
        row = process_one(p)
        per_sample_path = OUT_DIR / f"{p.stem}.jpg"
        row.save(per_sample_path, quality=90)
        print(f"  -> {per_sample_path}")
        rows.append(row)

    # Normalize row widths
    target_w = min(r.width for r in rows)
    norm = [r.resize((target_w, int(r.height * target_w / r.width)), Image.BILINEAR) for r in rows]
    total_h = sum(r.height for r in norm) + max(1, len(norm) - 1) * 8
    grid = Image.new("RGB", (target_w, total_h), (12, 12, 12))
    y = 0
    for r in norm:
        grid.paste(r, (0, y))
        y += r.height + 8

    out_path = OUT_DIR / "comparison.jpg"
    grid.save(out_path, quality=88)
    print(f"[test] wrote {out_path} ({grid.size})")

    # Also save a quick summary of per-image channel means
    print("\n[test] channel means (0-1):")
    print(f"  {'name':<20} {'orig RGB':<28} {'enhanced RGB':<28} {'wb RGB':<28}")
    for p in sources:
        img = load_and_resize(p)
        orig = np.asarray(img).astype(np.float32) / 255.0
        boxed, box = letterbox(img)
        enh = unletterbox(run_tflite(boxed), box, img.size[0], img.size[1])
        wb = gray_world_wb(enh)

        def m(a):
            return f"({a[..., 0].mean():.2f} {a[..., 1].mean():.2f} {a[..., 2].mean():.2f})"

        print(f"  {p.name:<20} {m(orig):<28} {m(enh):<28} {m(wb):<28}")


if __name__ == "__main__":
    main()
