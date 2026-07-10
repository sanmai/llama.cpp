#!/usr/bin/env python3
"""Inject per-expert ``*_exps.scale`` tensors into a self-quant N4_0 GGUF.

This turns a scale-free MoE self-quant into an artifact that exercises the
NVFP4 per-expert weight-scale fusion path (the ``mul_mat_id -> reshape ->
repeat -> get_rows -> mul`` epilogue) without needing a real NVFP4 checkpoint.

The injected scale is a deterministic per-expert ramp (not 1.0) so the
fused-vs-unfused bit-exact A/B genuinely tests per-expert scale indexing:
a misindexed scale would change the output, an all-ones scale would not.

Only ``*_exps.weight`` of type NVFP4 get a scale, so toggling
GGML_CUDA_DISABLE_SCALE_FUSION isolates the routed-expert fusion (attention,
SSM and shared-expert matmuls stay scale-free).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Prefer the in-tree gguf package (has NVFP4 == 40).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf  # noqa: E402


def ramp(n_expert: int) -> np.ndarray:
    # 0.75 .. 1.25, distinct per expert
    return (0.75 + 0.5 * np.arange(n_expert) / max(n_expert - 1, 1)).astype(np.float32)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <input.gguf> <output.gguf>")

    in_path, out_path = sys.argv[1], sys.argv[2]

    reader = gguf.GGUFReader(in_path, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()

    writer = gguf.GGUFWriter(out_path, arch=arch, endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    # copy key-value metadata verbatim (GGUFWriter rewrites ARCHITECTURE / GGUF.*)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    # build the per-expert scales for NVFP4 *_exps.weight tensors
    scales: list[tuple[str, np.ndarray]] = []
    for t in reader.tensors:
        if t.tensor_type == gguf.GGMLQuantizationType.NVFP4 and t.name.endswith("_exps.weight"):
            n_expert = int(t.shape[-1])
            scales.append((t.name[: -len(".weight")] + ".scale", ramp(n_expert)))

    print(f"injecting {len(scales)} *_exps.scale tensors", file=sys.stderr)

    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for name, arr in scales:
        writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    total = sum(t.n_bytes for t in reader.tensors) + sum(a.nbytes for _, a in scales)
    bar = tqdm(desc="Writing", total=total, unit="byte", unit_scale=True)
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
        bar.update(t.n_bytes)
    for _, arr in scales:
        writer.write_tensor_data(arr, tensor_endianess=reader.endianess)
        bar.update(arr.nbytes)
    writer.close()
    bar.close()


if __name__ == "__main__":
    main()
