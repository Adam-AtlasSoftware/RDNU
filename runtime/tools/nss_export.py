#!/usr/bin/env python3
"""
nss_export.py -- Arm NSS v1 backbone: weights + golden bundles for the RDNU DX12 engine.

Loads the released checkpoints (ml/models/nss, or explicit paths), runs the AutoEncoderV1
backbone on a fixed input, and writes .rdnut bundles the engine validates against:

  nss_backbone.rdnut       fp32 weights (OIHW) + per-layer golden        -> programNSS
  nss_backbone_int8.rdnut  INT8 weights (OHWI), scales, zero-points +
                           a float-requantised reference                  -> programNSSInt8

and the production artefacts consumed by the INT8 NHWC engine (runtime/src/engine):

  nss.rdnm                 binary manifest: tensors, layers, blob offsets
  nss.manifest.json        the same, human readable, with scales
  nss_w8.bin               weights: DP4a and OHWI int8 layouts, int32 bias, fp32 requant, LUTs
  nss_prod.rdnut           NHWC int8 input and the exact integer golden for every tensor

Quantisation scheme (torch.ao QAT as exported by the model gym):
  activations  per-tensor affine, zp = -128, q in [-128, 127]; one quantiser per tensor
               (concat inputs and upsample in/out share it; asserted)
  weights      per-output-channel symmetric, scale = max|w| / 127.5, q in [-128, 127]
  heads        conv -> logit quantiser (affine) -> sigmoid -> output quantiser (1/254, zp -127)

Production integer path (deterministic on every device: one int->float convert, one multiply,
round-half-even, integer clamp; no fused multiply-add anywhere):
  acc   = sum x_raw * wq + C[oc]          x_raw in [-128,127] including -128 padding
  C[oc] = 128 * sum wq[oc] + round(b[oc] / (s_in * s_w[oc]))
  relu  : q = clamp(round(f32(acc) * r[oc]) - 128, -128, 127),        r = s_in s_w / s_out
  head  : l = clamp(round(f32(acc) * r[oc]) + zp_logit, -128, 127),   r = s_in s_w / s_logit
          q = LUT[l + 128],  LUT[i] = clamp(round(sigmoid((i - 128 - zp_logit) s_logit) * 254) - 127)

The FX-traced INT8 checkpoint numbers its fake-quant modules in forward order; the table
below pins them to layers and is asserted against tensor shapes and the model-card scales.

Usage:
    python nss_export.py [--fp32 nss_v1_0_1_high_fp32.pt] [--int8 nss_v1_0_1_high_int8.pt]
                         [--h 64 --w 96] [--prod-h 72 --prod-w 104] [--seed 0]
                         [--out runtime/tools/golden] [--models runtime/models/NSS_INT8]
"""
import argparse
import importlib.util
import json
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
INPUT_SCALE = 0.003912401385605335   # model card: _PreprocessTensor (released high model)
OUTPUT_SCALE = 0.003937007859349251  # model card: _KpnCoefficients / _TemporalTensor

# Production graph (NHWC int8). Concats are channel ranges of one tensor, nearest upsample is a
# flag on the consumer. Columns: layer, in tensor, in channel offset, cin, stride, upsample-in,
# out tensor, out channel offset, cout, act, DP4a output-channel block.
PROD = [
    ("conv2d_0", "input", 0, 12, 2, 0, "c0", 0, 32, "relu", 16),
    ("conv2d_1", "c0", 0, 32, 1, 0, "cat9", 16, 32, "relu", 16),
    ("conv2d_2", "cat9", 16, 32, 2, 0, "c2", 0, 32, "relu", 16),
    ("conv2d_3", "c2", 0, 32, 1, 0, "cat7", 16, 32, "relu", 16),
    ("conv2d_4", "cat7", 16, 32, 2, 0, "c4", 0, 64, "relu", 16),
    ("conv2d_5", "c4", 0, 64, 1, 0, "c5", 0, 64, "relu", 16),
    ("conv2d_6", "c5", 0, 64, 1, 0, "c6", 0, 32, "relu", 16),
    ("conv2d_7", "c6", 0, 32, 1, 1, "cat7", 0, 16, "relu", 16),
    ("conv2d_8", "cat7", 0, 48, 1, 0, "c8", 0, 32, "relu", 16),
    ("kpn_params", "c8", 0, 32, 1, 0, "kpn", 0, 36, "sigmoid", 12),
    ("conv2d_9", "c8", 0, 32, 1, 1, "cat9", 0, 16, "relu", 16),
    ("conv2d_10", "cat9", 0, 48, 1, 0, "c10", 0, 16, "relu", 16),
    ("conv2d_11", "c10", 0, 16, 1, 0, "c11", 0, 16, "relu", 16),
    ("temporal_params_out_conv", "c11", 0, 16, 1, 1, "temporal", 0, 4, "sigmoid", 4),
]
# name, channels, level (resolution = input >> level), kind, quantiser module index
TENSORS = [
    ("input", 12, 0, "input", 0),
    ("c0", 32, 1, "arena", 3),
    ("cat9", 48, 1, "arena", 6),
    ("c2", 32, 2, "arena", 9),
    ("cat7", 48, 2, "arena", 12),
    ("c4", 64, 3, "arena", 15),
    ("c5", 64, 3, "arena", 18),
    ("c6", 32, 3, "arena", 21),
    ("c8", 32, 2, "arena", 29),
    ("c10", 16, 1, "arena", 41),
    ("c11", 16, 1, "arena", 44),
    ("kpn", 36, 2, "kpn", 33),
    ("temporal", 4, 0, "temporal", 49),
]
TENSOR_KIND = {"input": 0, "arena": 1, "kpn": 2, "temporal": 3}
# quantisers that must be identical (shared by concat/upsample edges)
SHARED_Q = [(21, 22), (25, 26), (12, 26), (6, 37), (37, 38), (29, 34), (44, 45)]
LOGIT_Q = {"kpn_params": 32, "temporal_params_out_conv": 48}
BLOB_ALIGN = 64
RDNM_VERSION = 1


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


def qparams(q, i):
    return float(q[f"activation_post_process_{i}.scale"].item()), int(q[f"activation_post_process_{i}.zero_point"].item())


def quantise_weights(w, ws):
    """torch fake-quant for weights: round-half-even, per-channel symmetric range [-128, 127]."""
    return np.clip(np.round(w / ws[:, None, None, None]), -128, 127)


def sigmoid_lut(s_logit, zp_logit):
    """LUT[i] for logit code i - 128, evaluated in float32 exactly as torch fake-quant does."""
    codes = torch.arange(-128, 128, dtype=torch.float32)
    sig = torch.sigmoid((codes - zp_logit) * torch.tensor(s_logit, dtype=torch.float32))
    out = torch.round(sig / torch.tensor(OUTPUT_SCALE, dtype=torch.float32)) - 127
    return np.clip(out.numpy(), -128, 127).astype(np.int8)


def prod_layers(q):
    """Per-layer integer parameters for the production path."""
    for a, b in SHARED_Q:
        assert qparams(q, a) == qparams(q, b), ("quantisers not shared", a, b)
    tq = {name: qparams(q, qi) for name, _, _, _, qi in TENSORS}
    for name, (s, zp) in tq.items():
        want = -127 if name in ("kpn", "temporal") else -128
        assert zp == want, (name, zp)
    layers = []
    for i, (n, tin, cin_off, cin, stride, up, tout, cout_off, cout, act, ocb) in enumerate(PROD):
        ai, wi = QUANT_IDX[n]
        s_in, zp_in = qparams(q, ai)
        assert (s_in, zp_in) == tq[tin], (n, "input quantiser differs from its tensor", s_in, tq[tin])
        w = q[f"_param_constant{2 * i}"].numpy().astype(np.float64)
        b = q[f"_param_constant{2 * i + 1}"].numpy().astype(np.float64)
        ws = q[f"activation_post_process_{wi}.scale"].numpy().astype(np.float64)
        assert (q[f"activation_post_process_{wi}.zero_point"].numpy() == 0).all(), n
        assert w.shape == (cout, cin, 3, 3), (n, w.shape)
        wq = quantise_weights(w, ws).astype(np.int64)
        acc_scale = s_in * ws
        C = 128 * wq.sum(axis=(1, 2, 3)) + np.round(b / acc_scale).astype(np.int64)
        L = {"name": n, "tin": tin, "cin_off": cin_off, "cin": cin, "stride": stride, "up": up,
             "tout": tout, "cout_off": cout_off, "cout": cout, "act": act, "ocb": ocb,
             "wq": wq, "C": C, "s_in": s_in, "wscale": ws, "bias": b}
        if act == "relu":
            s_out, _ = tq[tout]
            L["r"] = (acc_scale / s_out).astype(np.float32)
            L["s_out"] = s_out
        else:
            s_log, zp_log = qparams(q, LOGIT_Q[n])
            L["r"] = (acc_scale / s_log).astype(np.float32)
            L["zp_logit"] = zp_log
            L["s_logit"] = s_log
            L["lut"] = sigmoid_lut(s_log, zp_log)
        L["wmma"] = int(stride == 1 and not up and cin % 16 == 0 and cout % 16 == 0 and tin != "input")
        assert np.abs(C).max() < 2 ** 31
        layers.append(L)
    return layers, tq


def conv_raw(x_raw, wq, stride):
    """sum x_raw * wq over a 3x3 window with -128 padding; x_raw HWC int, wq OIHW int -> HWC int64."""
    t = torch.from_numpy(np.ascontiguousarray(x_raw.transpose(2, 0, 1)).astype(np.float64))[None]
    t = F.pad(t, (1, 1, 1, 1), value=-128.0)
    acc = F.conv2d(t, torch.from_numpy(wq.astype(np.float64)), None, stride, 0)[0]
    return acc.numpy().transpose(1, 2, 0).astype(np.int64)


def requant(acc, L):
    v = acc.astype(np.float32) * L["r"][None, None, :]
    v = np.round(v).astype(np.int64)
    if L["act"] == "relu":
        return np.clip(v - 128, -128, 127)
    logit = np.clip(v + L["zp_logit"], -128, 127)
    return L["lut"][logit + 128].astype(np.int64)


def prod_reference(x_raw, layers):
    """Integer golden on NHWC int8 tensors (logical H, W, C, no borders)."""
    Hp, Wp, _ = x_raw.shape
    level_dims = {l: (Hp >> l, Wp >> l) for l in range(4)}
    t = {"input": x_raw.astype(np.int64)}
    for name, ch, lvl, kind, _ in TENSORS[1:]:
        h, w = level_dims[lvl]
        t[name] = np.full((h, w, ch), -128, np.int64)
    for L in layers:
        src = t[L["tin"]][:, :, L["cin_off"]:L["cin_off"] + L["cin"]]
        if L["up"]:
            src = np.repeat(np.repeat(src, 2, 0), 2, 1)
        acc = conv_raw(src, L["wq"], L["stride"]) + L["C"][None, None, :]
        assert np.abs(acc).max() < 2 ** 31
        t[L["tout"]][:, :, L["cout_off"]:L["cout_off"] + L["cout"]] = requant(acc, L)
    return t


def fake_quant_check(ref, layers, tq):
    """Run every layer as the torch QAT graph does (dequantise, float conv + bias, fake-quantise)
    on the integer golden's own input to that layer, and compare codes. Layers are isolated so
    differences cannot compound; only the int32 bias rounding and float association differ."""
    worst = 0.0
    for L in layers:
        s_in, zp_in = tq[L["tin"]]
        src = (ref[L["tin"]][:, :, L["cin_off"]:L["cin_off"] + L["cin"]] - zp_in) * s_in
        if L["up"]:
            src = np.repeat(np.repeat(src, 2, 0), 2, 1)
        w = L["wq"] * L["wscale"][:, None, None, None]
        y = F.conv2d(torch.from_numpy(np.ascontiguousarray(src.transpose(2, 0, 1)))[None], torch.from_numpy(w),
                     torch.from_numpy(L["bias"]), L["stride"], 1)[0].numpy().transpose(1, 2, 0)
        if L["act"] == "relu":
            s, zp = tq[L["tout"]]
            code = np.clip(np.round(np.maximum(y, 0) / s) + zp, -128, 127)
            got = ref[L["tout"]][:, :, L["cout_off"]:L["cout_off"] + L["cout"]]
        else:
            # the sigmoid turns one logit code into up to ~11 output codes, so compare logits
            code = np.clip(np.round(y / L["s_logit"]) + L["zp_logit"], -128, 127)
            src_raw = ref[L["tin"]][:, :, L["cin_off"]:L["cin_off"] + L["cin"]]
            if L["up"]:
                src_raw = np.repeat(np.repeat(src_raw, 2, 0), 2, 1)
            acc = conv_raw(src_raw, L["wq"], L["stride"]) + L["C"][None, None, :]
            got = np.clip(np.round(acc.astype(np.float32) * L["r"][None, None, :]).astype(np.int64) + L["zp_logit"], -128, 127)
            assert (L["lut"][got + 128] == ref[L["tout"]][:, :, L["cout_off"]:L["cout_off"] + L["cout"]]).all()
        diff = np.abs(got - code)
        worst = max(worst, float(diff.max()))
        print(f"  {L['name']:26s} int vs fake-quant: {100 * float((diff == 0).mean()):7.3f}% identical, "
              f"max {int(diff.max())} step(s)")
    assert worst <= 1, "integer path diverges from the QAT graph by more than one code"


def dp4a_weights(wq, ocb):
    """[chunk][tap][cin/4][ocb] uint32, 4 input channels per word: one wave-uniform 16-word
    scalar load per (tap, channel group) in the DP4a kernel."""
    cout, cin = wq.shape[:2]
    assert cout % ocb == 0 and cin % 4 == 0
    w = wq.reshape(cout // ocb, ocb, cin // 4, 4, 9)          # chunk, j, g, k, tap
    w = w.transpose(0, 4, 2, 1, 3)                            # chunk, tap, g, j, k
    return np.ascontiguousarray(w).astype(np.int8)


def ohwi_weights(wq):
    return np.ascontiguousarray(wq.transpose(0, 2, 3, 1)).astype(np.int8)   # [cout][ky][kx][cin]


class Blob:
    def __init__(self):
        self.data = bytearray()

    def add(self, arr):
        pad = (-len(self.data)) % BLOB_ALIGN
        self.data += b"\0" * pad
        off = len(self.data)
        self.data += np.ascontiguousarray(arr).tobytes()
        return off


def write_production(layers, tq, x_raw, ref, out_dir, models_dir):
    blob = Blob()
    for L in layers:
        L["off_w_dp4a"] = blob.add(dp4a_weights(L["wq"], L["ocb"]))
        L["off_w_ohwi"] = blob.add(ohwi_weights(L["wq"]))
        L["off_bias"] = blob.add(L["C"].astype(np.int32))
        L["off_scale"] = blob.add(L["r"].astype(np.float32))
        L["off_lut"] = blob.add(L["lut"]) if "lut" in L else 0
    blob.data += b"\0" * ((-len(blob.data)) % BLOB_ALIGN)

    tidx = {name: i for i, (name, *_rest) in enumerate(TENSORS)}
    rdnm = bytearray(b"RDNM")
    rdnm += struct.pack("<7I", RDNM_VERSION, len(TENSORS), len(layers), len(blob.data), 0, 0, 0)
    for name, ch, lvl, kind, qi in TENSORS:
        s, zp = tq[name]
        rdnm += struct.pack("<16s4Iif2I", name.encode(), ch, lvl, TENSOR_KIND[kind], 0, zp, s, 0, 0)
    for L in layers:
        rdnm += struct.pack("<32s24I", L["name"].encode(),
                            tidx[L["tin"]], L["cin_off"], L["cin"], L["stride"], L["up"],
                            tidx[L["tout"]], L["cout_off"], L["cout"], int(L["act"] != "relu"), L["ocb"],
                            L["wmma"], L.get("zp_logit", 0) & 0xFFFFFFFF,
                            L["off_w_dp4a"], L["off_w_ohwi"], L["off_bias"], L["off_scale"], L["off_lut"],
                            0, 0, 0, 0, 0, 0, 0)
    manifest = {
        "version": RDNM_VERSION, "blob_bytes": len(blob.data),
        "tensors": [{"name": n, "channels": c, "level": l, "kind": k, "scale": tq[n][0], "zp": tq[n][1]}
                    for n, c, l, k, _ in TENSORS],
        "layers": [{k: (int(v) if isinstance(v, (np.integer, int)) and not isinstance(v, bool) else v)
                    for k, v in L.items() if k in ("name", "tin", "cin_off", "cin", "stride", "up", "tout",
                                                   "cout_off", "cout", "act", "ocb", "wmma", "zp_logit",
                                                   "s_in", "s_out", "s_logit", "off_w_dp4a", "off_w_ohwi",
                                                   "off_bias", "off_scale", "off_lut")}
                   for L in layers],
    }
    for d in (out_dir, models_dir):
        if not d:
            continue
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "nss.rdnm"), "wb").write(rdnm)
        open(os.path.join(d, "nss_w8.bin"), "wb").write(blob.data)
        with open(os.path.join(d, "nss.manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)

    bundle = {"input_nhwc": x_raw}
    for name, *_rest in TENSORS[1:]:
        bundle[("golden." if name in ("kpn", "temporal") else "chk.") + name] = ref[name]
    write_rdnut(os.path.join(out_dir, "nss_prod.rdnut"), bundle)
    print(f"wrote nss.rdnm ({len(rdnm)} B), nss_w8.bin ({len(blob.data)} B), nss_prod.rdnut "
          f"input {x_raw.shape} -> kpn {ref['kpn'].shape}, temporal {ref['temporal'].shape}")


def export_production(q, args):
    layers, tq = prod_layers(q)
    rng = np.random.default_rng(args.seed)
    x_raw = rng.integers(-128, 128, size=(args.prod_h, args.prod_w, 12))
    ref = prod_reference(x_raw, layers)
    fake_quant_check(ref, layers, tq)
    write_production(layers, tq, x_raw, ref, args.out, args.models)

    # dynamic resolution: a smaller frame inside the same allocation (top-left crop of the input)
    h2, w2 = args.prod_h - 16, args.prod_w - 16
    x2 = x_raw[:h2, :w2]
    ref2 = prod_reference(x2, layers)
    drs = {"input_nhwc": x2}
    for name, *_rest in TENSORS[1:]:
        drs[("golden." if name in ("kpn", "temporal") else "chk.") + name] = ref2[name]
    write_rdnut(os.path.join(args.out, "nss_prod_drs.rdnut"), drs)
    print(f"wrote nss_prod_drs.rdnut  input {x2.shape}")
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default=os.path.join(NSS_DIR, "nss_v1_0_1_high_fp32.pt"))
    ap.add_argument("--int8", default=os.path.join(NSS_DIR, "nss_v1_0_1_high_int8.pt"))
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=96)
    ap.add_argument("--prod-h", type=int, default=72)
    ap.add_argument("--prod-w", type=int, default=104)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "golden"))
    ap.add_argument("--models", default=os.path.join(ROOT, "runtime", "models", "NSS_INT8"),
                    help="where the shipped manifest + weight blob go ('' to skip)")
    args = ap.parse_args()
    assert args.h % 8 == 0 and args.w % 8 == 0, "input dims must be multiples of 8"
    assert args.prod_h % 8 == 0 and args.prod_w % 8 == 0, "input dims must be multiples of 8"
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
        wq = quantise_weights(w.astype(np.float64), ws)
        layers[n] = {"wq_oihw": wq.astype(np.float32), "wscale": ws, "ascale": a_s, "azp": a_z, "bias": b.astype(np.float64)}
    # the input scale is learned during QAT and travels in the manifest (rdnu_shaderc passes it
    # to the pre-process shader); the output scale is fixed by the sigmoid quantiser
    if abs(layers["conv2d_0"]["ascale"] - INPUT_SCALE) > 1e-9:
        print(f"input scale {layers['conv2d_0']['ascale']:.17g} differs from the released model's")
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

    export_production(q, args)


if __name__ == "__main__":
    main()
