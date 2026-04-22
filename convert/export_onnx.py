"""Export IAT enhance model to ONNX (fixed 512x512, opset 17).

Prereqs (run once from D:\\filterapp):
    git clone https://github.com/cuiziteng/Illumination-Adaptive-Transformer convert/iat_source
    # place best_Epoch_lol_v1.pth into convert/weights/

Run:
    python convert/export_onnx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
IAT_SRC = HERE / "iat_source" / "IAT_enhance"
WEIGHTS = HERE / "weights" / "best_Epoch_lol_v1.pth"
OUTPUT = HERE / "artifacts" / "iat_enhance.onnx"

if not IAT_SRC.exists():
    raise SystemExit(
        f"IAT source not found at {IAT_SRC}.\n"
        "Run: git clone https://github.com/cuiziteng/Illumination-Adaptive-Transformer "
        "convert/iat_source"
    )
sys.path.insert(0, str(IAT_SRC))

from model.IAT_main import IAT  # noqa: E402


class ExportWrapper(nn.Module):
    """Batch=1 rewrite of IAT.forward for clean onnx2tf conversion.

    The original IAT.forward uses a per-batch list comprehension with
    tensordot + clamp + pow that onnx2tf (flatbuffer_direct) mis-converts
    and that tf_converter still gets wrong on the 3x3 color matmul axis.
    We unroll the color matrix into explicit per-channel scalar multiplies,
    which every onnx2tf backend handles correctly.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.local_net = model.local_net
        self.global_net = model.global_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mul, add = self.local_net(x)
        img_high = x * mul + add  # (1, 3, H, W)

        gamma, color = self.global_net(x)  # gamma: (1,1), color: (1,3,3)

        r_in = img_high[:, 0:1]
        g_in = img_high[:, 1:2]
        b_in = img_high[:, 2:3]

        c = color[0]
        r_out = r_in * c[0, 0] + g_in * c[0, 1] + b_in * c[0, 2]
        g_out = r_in * c[1, 0] + g_in * c[1, 1] + b_in * c[1, 2]
        b_out = r_in * c[2, 0] + g_in * c[2, 1] + b_in * c[2, 2]
        img_colored = torch.cat([r_out, g_out, b_out], dim=1)
        img_colored = torch.clamp(img_colored, 1e-8, 1.0)

        return img_colored.pow(gamma[0, 0])


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(
            f"Weights not found at {WEIGHTS}.\n"
            "Download best_Epoch_lol_v1.pth from the IAT repo "
            "(IAT_enhance/best_Epoch_lol_v1.pth) into convert/weights/."
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    model = IAT()
    state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    wrapped = ExportWrapper(model).eval()
    dummy = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            str(OUTPUT),
            input_names=["input"],
            output_names=["enhanced"],
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"[export] wrote {OUTPUT}")

    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("[export] onnxsim not installed; skipping simplify step")
        return

    model_proto = onnx.load(str(OUTPUT))
    simplified, ok = simplify(model_proto)
    if not ok:
        print("[export] onnxsim did not converge; keeping original")
        return
    onnx.save(simplified, str(OUTPUT))
    print(f"[export] simplified {OUTPUT}")


if __name__ == "__main__":
    main()
