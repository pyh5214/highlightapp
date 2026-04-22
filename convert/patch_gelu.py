"""Replace Erf-based GELU with tanh approximation in ONNX model.

GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
       ≈ x * 0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))

This avoids FlexErf in TFLite, which requires the select-tf-ops library.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

HERE = Path(__file__).resolve().parent
SRC = HERE / "artifacts" / "iat_enhance.onnx"
DST = HERE / "artifacts" / "iat_enhance_patched.onnx"

SQRT_2_OVER_PI = 0.7978845608028654
COEFF = 0.044715


def find_gelu_patterns(graph):
    node_by_output = {o: n for n in graph.node for o in n.output}
    node_by_input = {}
    for n in graph.node:
        for i in n.input:
            node_by_input.setdefault(i, []).append(n)

    patterns = []
    for node in graph.node:
        if node.op_type != "Erf":
            continue
        erf_in = node.input[0]
        erf_out = node.output[0]
        if erf_in not in node_by_output:
            continue
        div_node = node_by_output[erf_in]
        if div_node.op_type != "Div":
            continue
        x = div_node.input[0]

        if erf_out not in node_by_input:
            continue
        add_nodes = [n for n in node_by_input[erf_out] if n.op_type == "Add"]
        if not add_nodes:
            continue
        add_node = add_nodes[0]
        add_out = add_node.output[0]

        if add_out not in node_by_input:
            continue
        mul_half_nodes = [n for n in node_by_input[add_out] if n.op_type == "Mul"]
        if not mul_half_nodes:
            continue
        mul_half = mul_half_nodes[0]
        mul_half_out = mul_half.output[0]

        if mul_half_out not in node_by_input:
            continue
        mul_x_nodes = [n for n in node_by_input[mul_half_out] if n.op_type == "Mul"]
        if not mul_x_nodes:
            continue
        mul_x = mul_x_nodes[0]
        gelu_out = mul_x.output[0]

        patterns.append({
            "x": x,
            "gelu_out": gelu_out,
            "trigger_name": div_node.name,  # insert replacement here
            "names_to_remove": {div_node.name, node.name, add_node.name,
                                 mul_half.name, mul_x.name},
        })
        print(f"  GELU: {x!r} -> {gelu_out!r}")

    return patterns


def make_tanh_gelu_nodes(x: str, gelu_out: str, suffix: str):
    def uid(s):
        return f"_gelu_{suffix}_{s}"

    nodes = []
    inits = []

    pow_exp = numpy_helper.from_array(np.array(3.0, dtype=np.float32), name=uid("pow_exp"))
    inits.append(pow_exp)
    pow_out = uid("x_cube")
    nodes.append(helper.make_node("Pow", [x, uid("pow_exp")], [pow_out]))

    coeff = numpy_helper.from_array(np.array(COEFF, dtype=np.float32), name=uid("coeff"))
    inits.append(coeff)
    coeff_out = uid("coeff_x3")
    nodes.append(helper.make_node("Mul", [uid("coeff"), pow_out], [coeff_out]))

    inner_out = uid("inner")
    nodes.append(helper.make_node("Add", [x, coeff_out], [inner_out]))

    kappa = numpy_helper.from_array(np.array(SQRT_2_OVER_PI, dtype=np.float32), name=uid("kappa"))
    inits.append(kappa)
    scaled_out = uid("scaled")
    nodes.append(helper.make_node("Mul", [uid("kappa"), inner_out], [scaled_out]))

    tanh_out = uid("tanh")
    nodes.append(helper.make_node("Tanh", [scaled_out], [tanh_out]))

    one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=uid("one"))
    inits.append(one)
    add_out = uid("add1")
    nodes.append(helper.make_node("Add", [uid("one"), tanh_out], [add_out]))

    half = numpy_helper.from_array(np.array(0.5, dtype=np.float32), name=uid("half"))
    inits.append(half)
    half_out = uid("half_mul")
    nodes.append(helper.make_node("Mul", [uid("half"), add_out], [half_out]))

    nodes.append(helper.make_node("Mul", [x, half_out], [gelu_out]))

    return nodes, inits


def patch(src: Path, dst: Path):
    model = onnx.load(str(src))
    graph = model.graph

    patterns = find_gelu_patterns(graph)
    if not patterns:
        print("No Erf-based GELU patterns found.")
        return

    # Map trigger_name -> replacement nodes/inits
    replacements = {}
    extra_inits = []
    for i, p in enumerate(patterns):
        nodes, inits = make_tanh_gelu_nodes(p["x"], p["gelu_out"], str(i))
        replacements[p["trigger_name"]] = nodes
        extra_inits.extend(inits)

    all_names_to_remove = set()
    for p in patterns:
        all_names_to_remove |= p["names_to_remove"]

    new_nodes = []
    for node in graph.node:
        if node.name in replacements:
            new_nodes.extend(replacements[node.name])
        elif node.name not in all_names_to_remove:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(extra_inits)

    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    print(f"Patched {len(patterns)} GELU(s) -> {dst.name}")


if __name__ == "__main__":
    print(f"Patching {SRC.name} ...")
    patch(SRC, DST)
