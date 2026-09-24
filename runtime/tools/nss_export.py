#!/usr/bin/env python3
"""
nss_export.py -- Arm NSS v1 backbone: weights + golden bundles for the RDNU DX12 engine.

Loads the released checkpoints (ml/models/nss, or explicit paths), runs the AutoEncoderV1
backbone on a fixed input, and writes .rdnut bundles the engine validates against:

  nss_backbone.rdnut       fp32 weights (OIHW) + per-layer golden        -> programNSS
  nss_backbone_int8.rdnut  INT8 weights (OHWI), scales, zero-points +
                           the exact integer-path reference               -> programNSSInt8

Quantisation scheme (torch.ao QAT as exported by the model gym):
  activations  per-tensor affine, zp = -128 (post-ReLU ranges), q in [-128, 127]
               xq = clamp(round(x / s_in) + zp, -128, 127)
  weights      per-output-channel symmetric, q in [-127, 127]
  conv         acc = sum (xq - zp) * wq   (int32);  y = acc * s_in * s_w[oc] + b[oc]; ReLU / sigmoid
  head outputs sigmoid, then a fixed quantiser (scale 1/254, zp -127) in the VGF graph; the
               runtime consumes the sigmoid output directly, so the golden stops there.

The FX-traced INT8 checkpoint numbers its fake-quant modules in forward order; the table
below pins them to layers and is asserted against tensor shapes and the model-card scales.

Usage:
    python nss_export.py [--fp32 nss_v1_0_1_high_fp32.pt] [--int8 nss_v1_0_1_high_int8.pt]
                         [--h 64 --w 96] [--seed 0] [--out runtime/tools/golden]
"""
import argparse
import importlib.util
import os
import struct
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GYM = os.path.join(ROOT, "ml", "model-gym", "src", "ng_model_gym")
NSS_DIR = os.path.join(ROOT, "ml", "models", "nss")

# (layer, stride, activation): forward order of AutoEncoderV1
LAYERS = [
    ("conv2d_0", 2, "relu"), ("conv2d_1", 1, "relu"),
    ("conv2d_2", 2, "relu"), ("conv2d_3", 1, "relu"),
    ("conv2d_4", 2, "relu"), ("conv2d_5", 1, "relu"),
    ("conv2d_6", 1, "relu"),
    ("conv2d_7", 1, "relu"), ("conv2d_8", 1, "relu"),
    ("kpn_params", 1, "sigmoid"),
    ("conv2d_9", 1, "relu"), ("conv2d_10", 1, "relu"), ("conv2d_11", 1, "relu"),
    ("temporal_params_out_conv", 1, "sigmoid"),
]
# layer -> (input activation quantiser, weight quantiser) module index in the INT8 checkpoint
QUANT_IDX = {
    "conv2d_0": (0, 1), "conv2d_1": (3, 4), "conv2d_2": (6, 7), "conv2d_3": (9, 10),
    "conv2d_4": (12, 13), "conv2d_5": (15, 16), "conv2d_6": (18, 19),
    "conv2d_7": (22, 23), "conv2d_8": (26, 27), "kpn_params": (29, 30),
    "conv2d_9": (34, 35), "conv2d_10": (38, 39), "conv2d_11": (41, 42),
    "temporal_params_out_conv": (45, 46),
}
INPUT_SCALE = 0.003912401385605335   # model card: _PreprocessTensor
OUTPUT_SCALE = 0.003937007859349251  # model card: _KpnCoefficients / _TemporalTensor


def load_backbone_class():
    """Import AutoEncoderV1 from the gym submodule without installing the package."""
    def mod(name):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
        return m

    for n in ["ng_model_gym", "ng_model_gym.core", "ng_model_gym.core.model", "ng_model_gym.core.model.layers"]:
        mod(n)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    load("ng_model_gym.core.model.layers.conv_block", os.path.join(GYM, "core", "model", "layers", "conv_block.py"))
    blocks = load("nss_model_blocks_v1", os.path.join(GYM, "usecases", "nss", "model", "model_blocks_v1.py"))
    return blocks.AutoEncoderV1


def load_checkpoint(path):
    if os.path.getsize(path) < 4096:
        sys.exit(f"{path} is a Git LFS pointer, not the checkpoint. Pull it "
                 f"(git lfs pull in ml/models/nss) or pass --fp32/--int8 with a real file.")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model_state_dict"]


def write_rdnut(path, tensors):
    """Same format as dump_golden.py: 'RDNT', version, count, then name/dims/float32 data."""
    with open(path, "wb") as f:
        f.write(b"RDNT")
        f.write(struct.pack("<II", 1, len(tensors)))
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb + struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", int(d)))
            f.write(arr.tobytes())


def q_round(v):
    return np.floor(v + 0.5)   # round-half-up, identical to the HLSL kernels


def int8_reference(x, layers):
    """Integer-path forward in float64 (exact): returns dict of post-activation outputs."""
    out, acts = {}, {}
    cur = x.astype(np.float64)
    skip = {}
    for name, stride, act in LAYERS:
        L = layers[name]
        if name == "conv2d_7":
            cur = np.repeat(np.repeat(acts["conv2d_6"], 2, 1), 2, 2)
        elif name == "conv2d_8":
            cur = np.concatenate([acts["conv2d_7"], acts["conv2d_3"]], 0)
        elif name == "kpn_params":
            cur = acts["conv2d_8"]
        elif name == "conv2d_9":
            cur = np.repeat(np.repeat(acts["conv2d_8"], 2, 1), 2, 2)
        elif name == "conv2d_10":
            cur = np.concatenate([acts["conv2d_9"], acts["conv2d_1"]], 0)
        elif name == "temporal_params_out_conv":
            cur = np.repeat(np.repeat(acts["conv2d_11"], 2, 1), 2, 2)
        s_in, zp = L["ascale"], L["azp"]
        xq = np.clip(q_round(cur / s_in) + zp, -128, 127) - zp
        wq = torch.from_numpy(L["wq_oihw"].astype(np.float64))
        acc = F.conv2d(torch.from_numpy(xq)[None], wq, None, stride, 1)[0].numpy()
        y = acc * s_in * L["wscale"][:, None, None] + L["bias"][:, None, None]
        y = np.maximum(y, 0.0) if act == "relu" else 1.0 / (1.0 + np.exp(-y))
        acts[name] = y
        out[name] = y.astype(np.float32)
        cur = y
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default=os.path.join(NSS_DIR, "nss_v1_0_1_high_fp32.pt"))
    ap.add_argument("--int8", default=os.path.join(NSS_DIR, "nss_v1_0_1_high_int8.pt"))
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "golden"))
    args = ap.parse_args()
    assert args.h % 8 == 0 and args.w % 8 == 0, "input dims must be multiples of 8"
    os.makedirs(args.out, exist_ok=True)

    AutoEncoderV1 = load_backbone_class()
    model = AutoEncoderV1(batch_norm=False, kpn_size=(6, 6)).eval()
    sd = {k[len("autoencoder."):]: v for k, v in load_checkpoint(args.fp32).items()}
    model.load_state_dict(sd, strict=True)

    torch.manual_seed(args.seed)
    x = torch.rand(1, 12, args.h, args.w)

    acts = {}
    hooks = [getattr(model, n).register_forward_hook(lambda m, i, o, n=n: acts.__setitem__(n, o[0].detach().numpy()))
             for n, _, _ in LAYERS]
    with torch.no_grad():
        kpn, temporal = model(x)
    for h in hooks:
        h.remove()

    macs = params = 0
    scale = {"conv2d_0": 4}
    res = 1.0
    for n, stride, _ in LAYERS:
        w = sd[n + ".conv2d.weight"]
        cout, cin, kh, kw = w.shape
        params += w.numel() + cout
        if n in ("conv2d_7", "conv2d_9", "temporal_params_out_conv"):
            res *= 4
        if stride == 2:
            res /= 4
        macs += cout * cin * kh * kw * res
    print(f"backbone: {params} params, {macs:.0f} MAC per input pixel, "
          f"{macs / 4:.0f} per output pixel at 2x")

    fp = {"input": x[0].numpy(), "golden_out": temporal[0].numpy(), "chk.kpn": kpn[0].numpy()}
    for n, _, _ in LAYERS:
        fp[n + ".weight"] = sd[n + ".conv2d.weight"].numpy()
        fp[n + ".bias"] = sd[n + ".conv2d.bias"].numpy()
    for n in ("conv2d_1", "conv2d_3", "conv2d_6", "conv2d_8", "conv2d_11"):
        fp["chk." + n] = acts[n]
    write_rdnut(os.path.join(args.out, "nss_backbone.rdnut"), fp)
    print(f"wrote nss_backbone.rdnut  input {tuple(x.shape[1:])} -> temporal {tuple(temporal.shape[1:])}, "
          f"kpn {tuple(kpn.shape[1:])}")

    if not os.path.exists(args.int8):
        print("no INT8 checkpoint, skipping INT8 bundle")
        return
    q = {k[len("autoencoder."):]: v for k, v in load_checkpoint(args.int8).items()}
    layers = {}
    for i, (n, _, _) in enumerate(LAYERS):
        ai, wi = QUANT_IDX[n]
        w = q[f"_param_constant{2 * i}"].numpy()
        b = q[f"_param_constant{2 * i + 1}"].numpy()
        ws = q[f"activation_post_process_{wi}.scale"].numpy().astype(np.float64)
        wz = q[f"activation_post_process_{wi}.zero_point"].numpy()
        a_s = float(q[f"activation_post_process_{ai}.scale"].item())
        a_z = int(q[f"activation_post_process_{ai}.zero_point"].item())
        assert w.shape == tuple(sd[n + ".conv2d.weight"].shape), n
        assert ws.shape == (w.shape[0],) and (wz == 0).all(), n
        assert a_z == -128, (n, a_z)
        wq = np.clip(q_round(w / ws[:, None, None, None]), -127, 127)
        layers[n] = {"wq_oihw": wq.astype(np.float32), "wscale": ws, "ascale": a_s, "azp": a_z, "bias": b.astype(np.float64)}
    assert abs(layers["conv2d_0"]["ascale"] - INPUT_SCALE) < 1e-9
    assert abs(float(q["activation_post_process_33.scale"].item()) - OUTPUT_SCALE) < 1e-9

    ref = int8_reference(x[0].numpy(), layers)
    i8 = {"input": x[0].numpy(), "golden_out": ref["temporal_params_out_conv"], "chk.kpn": ref["kpn_params"]}
    for n in ("conv2d_1", "conv2d_3", "conv2d_6", "conv2d_8", "conv2d_11"):
        i8["chk." + n] = ref[n]
    for n, L in layers.items():
        i8[n + ".weight.w"] = np.ascontiguousarray(np.transpose(L["wq_oihw"], (0, 2, 3, 1)))   # OHWI
        i8[n + ".weight.scale"] = L["wscale"].astype(np.float32)
        i8[n + ".weight.ascale"] = np.array([L["ascale"], L["azp"], -128, 127], np.float32)
        i8[n + ".weight.bias"] = L["bias"].astype(np.float32)
    write_rdnut(os.path.join(args.out, "nss_backbone_int8.rdnut"), i8)

    d = np.abs(ref["temporal_params_out_conv"] - temporal[0].numpy()).max()
    dk = np.abs(ref["kpn_params"] - kpn[0].numpy()).max()
    print(f"wrote nss_backbone_int8.rdnut  (QAT int8 vs fp32 checkpoint on this input: "
          f"max|dtemporal| {d:.3g}, max|dkpn| {dk:.3g}; these differ by design, both are goldens)")


if __name__ == "__main__":
    main()
