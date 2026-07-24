#!/usr/bin/env python3
"""
dump_golden.py -- RDNU golden-reference generator (Phase 0 validation harness).

Reconstructs the RDG model from the exported FP16 weights (or a .pth if supplied),
runs it on a fixed deterministic input, and dumps every layer's output tensor as the
"golden" reference that the DX12/WMMA engine is validated against, layer by layer.

No basicsr install is required: the architecture modules in RDG/ are loaded directly
with lightweight stubs for the few basicsr / torchvision symbols they import.

Usage:
    python dump_golden.py [--ckpt net_g.pth] [--res 64] [--frames 2] [--seed 0]

Model config is pinned by the exported weights: num_feat=36, scale=2 (x2 network),
enc/dec blocks [1,1], middle 1. The FP16 weight blob is stored OIHW (verified against
the INT8 model).
"""
import os
import re
import sys
import types
import argparse
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RDG_DIR = os.path.join(ROOT, "RDG")
FP16_DIR = os.path.join(ROOT, "runtime", "models", "RDNU_FP16")
OUT_DIR = os.path.join(os.path.dirname(__file__), "golden")


# --------------------------------------------------------------------- stubs
# The RDG arch imports a handful of basicsr / torchvision symbols. Rather than
# installing the full (heavy, Python-3.8-era) basicsr package, inject minimal stubs.
def _mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


_mod("basicsr")
_mod("basicsr.utils")
_mod("basicsr.archs")

_reg = _mod("basicsr.utils.registry")


class _Registry:
    def register(self, *a, **k):
        def deco(cls):
            return cls
        return deco


_reg.ARCH_REGISTRY = _Registry()


def motion_warp(x, motion, interp_mode="bilinear", padding_mode="zeros", align_corners=True):
    # Verbatim from basicsr/utils/render_data_util.py (pure torch, no cv2 dependency).
    _, _, h, w = x.size()
    if x.size()[-2:] != motion.size()[-2:]:
        motion = F.interpolate(motion, size=(h, w), mode="bilinear")
    motion = motion.permute(0, 2, 3, 1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x), indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    vgrid = grid + motion
    vgx = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgy = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vg = torch.stack((vgx, vgy), dim=3)
    return F.grid_sample(x, vg, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)


_rd = _mod("basicsr.utils.render_data_util")
_rd.motion_warp = motion_warp

# torchvision.ops.deform_conv is imported by rdg_block but unused at inference time.
_mod("torchvision")
_mod("torchvision.ops")
_dc = _mod("torchvision.ops.deform_conv")
_dc.deform_conv2d = lambda *a, **k: None


import importlib.util  # noqa: E402


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m


_load("basicsr.archs.rdg_block", os.path.join(RDG_DIR, "basicsr", "archs", "rdg_block.py"))
rdg_arch = _load("basicsr.archs.rdg_arch", os.path.join(RDG_DIR, "basicsr", "archs", "rdg_arch.py"))


# ------------------------------------------------------------------- weights
def load_fp16_state_dict():
    """Reconstruct the PyTorch state_dict from the exported FP16 blob (OIHW)."""
    meta = open(os.path.join(FP16_DIR, "rdnu_weights_meta.h")).read()
    blob = open(os.path.join(FP16_DIR, "rdnu_weights.bin"), "rb").read()
    entries = re.findall(r'\{"([^"]+)",\s*(\d+),\s*(\d+),\s*\{([^}]*)\},\s*(\d+)\}', meta)
    sd = {}
    for name, off, size, dims_s, nd in entries:
        off, size, nd = int(off), int(size), int(nd)
        dims = [int(x) for x in dims_s.split(",") if x.strip() != ""][:nd]
        arr = np.frombuffer(blob, np.float16, size // 2, off).astype(np.float32)
        sd[name] = torch.from_numpy(arr.copy()).reshape(dims)
    return sd


def write_rdnut(path, tensors):
    """Write a simple self-contained tensor bundle the DX12 harness can read.

    Format (little-endian): 'RDNT', u32 version=1, u32 count, then per tensor:
    u32 name_len, name bytes, u32 ndim, u32 dims[ndim], float32 data[prod(dims)].
    """
    import struct
    with open(path, "wb") as f:
        f.write(b"RDNT")
        f.write(struct.pack("<II", 1, len(tensors)))
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
            nb = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", int(d)))
            f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="optional net_g_*.pth (FP32 reference); default reconstructs from FP16")
    ap.add_argument("--res", type=int, default=64, help="LR resolution (square)")
    ap.add_argument("--frames", type=int, default=2, help="temporal frames")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = rdg_arch.RDG(in_channle=3, num_feat=36, scale=2,
                         middle_blk_num=1, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])
    model.eval()

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        raw = ck.get("params", ck)
        sd = {k.replace("module.", ""): v.float() for k, v in raw.items()}
        src = f"checkpoint {args.ckpt}"
    else:
        sd = load_fp16_state_dict()
        src = "FP16 export (reconstructed)"

    # Load only shape-matching tensors; report anything off so layout bugs surface loudly.
    model_sd = model.state_dict()
    ok, mismatch = {}, []
    for k, v in sd.items():
        if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape):
            ok[k] = v
        elif k in model_sd:
            mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
    model.load_state_dict(ok, strict=False)
    missing = [k for k in model_sd if k not in ok]
    unexpected = [k for k in sd if k not in model_sd]

    print(f"Loaded weights from {src}")
    print(f"  model tensors    : {len(model_sd)}")
    print(f"  provided tensors : {len(sd)}   matched: {len(ok)}")
    print(f"  missing keys     : {len(missing)}  {missing[:6]}")
    print(f"  unexpected keys  : {len(unexpected)}  {unexpected[:6]}")
    print(f"  shape mismatches : {len(mismatch)}  {mismatch[:6]}")

    # Deterministic input (ranges roughly match training normalization: [0,1], small motion).
    b, t, r = 1, args.frames, args.res

    def rnd(c):
        return torch.rand(b, t, c, r, r)

    inp = {
        "Image": rnd(3),
        "Depth": rnd(1),
        "Normal": rnd(3),
        "BRDF": rnd(3),
        "Motion": torch.randn(b, t, 2, r, r) * 2.0,
    }

    # Capture every submodule's input[0] and output tensor. Inputs let us build a
    # self-contained .rdnut bundle for ANY layer (its input feeds the kernel, its
    # output is the golden). Both are captured on the same (final) frame, so they stay
    # a consistent input->output pair for that layer.
    golden = {}
    golden_in = {}    # input[0] of each module (the primary feature input)
    golden_in1 = {}   # input[1] where present (HFB's g-buffer, CTR/DecodeLayer's g)
    golden_in2 = {}   # input[2] where present (DecodeLayer's depth)
    # First-frame (first call) captures. CTR is recurrent: on frame 0 pre_h is None, so it
    # self-attends with an identity motion-warp — the case to validate before grid-sample exists.
    golden_f = {}
    golden_f_in = {}
    golden_f_in1 = {}
    golden_f_in2 = {}
    handles = []
    for name, mod in model.named_modules():
        if name == "":
            continue

        def hook(m, i, o, nm=name):
            if isinstance(o, torch.Tensor):
                golden[nm] = o.detach().float().cpu().numpy()
                golden_f.setdefault(nm, golden[nm])
            if len(i) and isinstance(i[0], torch.Tensor):
                golden_in[nm] = i[0].detach().float().cpu().numpy()
                golden_f_in.setdefault(nm, golden_in[nm])
            if len(i) > 1 and isinstance(i[1], torch.Tensor):
                golden_in1[nm] = i[1].detach().float().cpu().numpy()
                golden_f_in1.setdefault(nm, golden_in1[nm])
            if len(i) > 2 and isinstance(i[2], torch.Tensor):
                golden_in2[nm] = i[2].detach().float().cpu().numpy()
                golden_f_in2.setdefault(nm, golden_in2[nm])

        handles.append(mod.register_forward_hook(hook))

    with torch.no_grad():
        out = model(inp)
    for h in handles:
        h.remove()

    os.makedirs(OUT_DIR, exist_ok=True)
    save = {f"in.{k}": v.numpy() for k, v in inp.items()}
    save["output"] = out.detach().numpy()
    save.update({f"act.{k}": v for k, v in golden.items()})
    np.savez(os.path.join(OUT_DIR, "golden.npz"), **save)

    # --- self-contained per-layer validation bundles for the DX12 harness ---
    # Any nn.Conv2d layer can be turned into a bundle: its captured input feeds the
    # kernel, its parameters + a small conv-params record drive the general conv2d
    # shader, and its captured output is the golden. batch dim (0) is dropped.
    mods = dict(model.named_modules())

    def emit_conv_bundle(fname, name):
        m = mods[name]
        (kh, kw) = m.kernel_size
        (sh, sw) = m.stride
        (ph, pw) = m.padding
        groups = m.groups
        x = golden_in[name][0]                       # (Cin, H, W)
        y = golden[name][0]                          # (Cout, OH, OW)
        w = ok[name + ".weight"].numpy()             # (Cout, Cin/groups, kh, kw) OIHW
        bkey = name + ".bias"
        b = ok[bkey].numpy() if bkey in ok else np.zeros(w.shape[0], np.float32)
        # params record (float32): [KH, KW, PadH, PadW, StrideH, StrideW, Groups]
        params = np.array([kh, kw, ph, pw, sh, sw, groups], dtype=np.float32)
        write_rdnut(os.path.join(OUT_DIR, fname), {
            "input": x, "weight": w, "bias": b, "golden_out": y, "params": params,
        })
        print(f"  bundle {fname:<24} {name:<28} "
              f"in{tuple(x.shape)} w{tuple(w.shape)} -> out{tuple(y.shape)}  "
              f"k{kh}x{kw} s{sh}x{sw} p{ph}x{pw} g{groups}")

    def emit_shift_bundle(fname, name):
        # CycleFC's fixed depthwise "shift" conv (groups=in, non-square 1x7/7x1 kernel).
        # It is an F.conv2d call, not a submodule, so we compute its golden directly with
        # torch (still a real reference) using the module's registered shift_weight buffer.
        m = mods[name]
        (kh, kw) = m.kernel_size
        pad_y, pad_x = kh // 2, kw // 2
        xt = torch.from_numpy(golden_in[name])       # (b, Cin, H, W)
        sw = m.shift_weight.detach()                 # (Cin, 1, kh, kw)
        yt = F.conv2d(xt, sw, padding=(pad_y, pad_x), groups=m.in_channels)
        params = np.array([kh, kw, pad_y, pad_x, 1, 1, m.in_channels], dtype=np.float32)
        write_rdnut(os.path.join(OUT_DIR, fname), {
            "input": xt[0].numpy(), "weight": sw.numpy(),
            "bias": np.zeros(m.in_channels, np.float32),
            "golden_out": yt[0].numpy(), "params": params,
        })
        print(f"  bundle {fname:<24} {name:<28} "
              f"in{tuple(xt.shape[1:])} w{tuple(sw.shape)} -> out{tuple(yt.shape[1:])}  "
              f"k{kh}x{kw} p{pad_y}x{pad_x} g{m.in_channels} (shift)")

    def emit_act_bundle(fname, name):
        # Pointwise activation (no params): captured input -> captured output.
        x = golden_in[name][0]
        y = golden[name][0]
        write_rdnut(os.path.join(OUT_DIR, fname), {"input": x, "golden_out": y})
        print(f"  bundle {fname:<24} {name:<28} in{tuple(x.shape)} -> out{tuple(y.shape)} (act)")

    def dfm_tensors(dfm_name, prefix):
        # All DFM weights, keyed `<prefix><sub>` to match the engine's program. shift_weight is a
        # registered buffer (deterministic, not a learned param) → pulled from the live module.
        d = mods[dfm_name]
        return {
            prefix + "proj_in.0.weight":   ok[dfm_name + ".proj_in.0.weight"].numpy(),
            prefix + "proj_in.0.bias":     ok[dfm_name + ".proj_in.0.bias"].numpy(),
            prefix + "proj_in.1.weight":   ok[dfm_name + ".proj_in.1.weight"].numpy(),
            prefix + "proj_in.1.bias":     ok[dfm_name + ".proj_in.1.bias"].numpy(),
            prefix + "mlp_h.shift_weight": d.mlp_h.shift_weight.detach().numpy(),
            prefix + "mlp_h.weight":       ok[dfm_name + ".mlp_h.weight"].numpy(),
            prefix + "mlp_h.bias":         ok[dfm_name + ".mlp_h.bias"].numpy(),
            prefix + "mlp_w.shift_weight": d.mlp_w.shift_weight.detach().numpy(),
            prefix + "mlp_w.weight":       ok[dfm_name + ".mlp_w.weight"].numpy(),
            prefix + "mlp_w.bias":         ok[dfm_name + ".mlp_w.bias"].numpy(),
            prefix + "merge.weight":       ok[dfm_name + ".merge.weight"].numpy(),
            prefix + "merge.bias":         ok[dfm_name + ".merge.bias"].numpy(),
            prefix + "proj_out.weight":    ok[dfm_name + ".proj_out.weight"].numpy(),
            prefix + "proj_out.bias":      ok[dfm_name + ".proj_out.bias"].numpy(),
        }

    def ccm_tensors(ccm_name, prefix):
        return {
            prefix + "fn.0.weight": ok[ccm_name + ".fn.0.weight"].numpy(),
            prefix + "fn.0.bias":   ok[ccm_name + ".fn.0.bias"].numpy(),
            prefix + "fn.2.weight": ok[ccm_name + ".fn.2.weight"].numpy(),
            prefix + "fn.2.bias":   ok[ccm_name + ".fn.2.bias"].numpy(),
        }

    def hfb_weights(hfb_name, prefix):
        return {
            prefix + "conv_g.weight":  ok[hfb_name + ".conv_g.weight"].numpy(),
            prefix + "conv_g.bias":    ok[hfb_name + ".conv_g.bias"].numpy(),
            prefix + "conv_x.weight":  ok[hfb_name + ".conv_x.weight"].numpy(),
            prefix + "conv_x.bias":    ok[hfb_name + ".conv_x.bias"].numpy(),
            prefix + "proj_out.weight": ok[hfb_name + ".proj_out.weight"].numpy(),
            prefix + "proj_out.bias":   ok[hfb_name + ".proj_out.bias"].numpy(),
        }

    def hfb_tensors(hfb_name, prefix):
        # HFB's second feature input is the g-buffer (module input[1]); the engine resizes it
        # to x's size, which is an identity when they already match (as at decoders.1, 64x64).
        return {prefix + "g": golden_in1[hfb_name][0], **hfb_weights(hfb_name, prefix)}

    def ctr_weights(ctr_name, prefix):
        return {
            prefix + "to_q.0.weight":  ok[ctr_name + ".to_q.0.weight"].numpy(),
            prefix + "to_q.0.bias":    ok[ctr_name + ".to_q.0.bias"].numpy(),
            prefix + "to_q.1.weight":  ok[ctr_name + ".to_q.1.weight"].numpy(),
            prefix + "to_q.1.bias":    ok[ctr_name + ".to_q.1.bias"].numpy(),
            prefix + "to_kv.0.weight": ok[ctr_name + ".to_kv.0.weight"].numpy(),
            prefix + "to_kv.0.bias":   ok[ctr_name + ".to_kv.0.bias"].numpy(),
            prefix + "to_kv.1.weight": ok[ctr_name + ".to_kv.1.weight"].numpy(),
            prefix + "to_kv.1.bias":   ok[ctr_name + ".to_kv.1.bias"].numpy(),
            prefix + "proj_out.weight": ok[ctr_name + ".proj_out.weight"].numpy(),
            prefix + "proj_out.bias":   ok[ctr_name + ".proj_out.bias"].numpy(),
        }

    def ctr_tensors(ctr_name, prefix):
        # CTR's second feature input is depth (module input[1]); frame-0 (identity-warp) values.
        return {prefix + "depth": golden_f_in1[ctr_name][0], **ctr_weights(ctr_name, prefix)}

    def emit_block_bundle(fname, name, tensors, desc, first=False):
        # first=True uses the frame-0 (first-call) captures — needed for CTR's identity-warp case.
        gi, go = (golden_f_in, golden_f) if first else (golden_in, golden)
        x = gi[name][0]
        y = go[name][0]
        write_rdnut(os.path.join(OUT_DIR, fname), {"input": x, "golden_out": y, **tensors})
        print(f"  bundle {fname:<24} {name:<28} in{tuple(x.shape)} -> out{tuple(y.shape)} ({desc})")

    print("wrote layer bundles:")
    emit_conv_bundle("first_conv.rdnut", "first_conv")                  # 3x3, 3->36
    emit_conv_bundle("ffn_conv3x3.rdnut", "encoders.0.0.ffn.fn.0")      # 3x3, 36->72
    emit_conv_bundle("ffn_conv1x1.rdnut", "encoders.0.0.ffn.fn.2")      # 1x1, 72->36
    emit_conv_bundle("dfm_dw3x3.rdnut", "encoders.0.0.dfm.proj_in.0")   # 3x3 g36, 36->72 (depthwise)
    emit_shift_bundle("cyclefc_shift_1x7.rdnut", "encoders.0.0.dfm.mlp_h")  # 1x7 shift, g36
    emit_shift_bundle("cyclefc_shift_7x1.rdnut", "encoders.0.0.dfm.mlp_w")  # 7x1 shift, g36
    emit_act_bundle("gelu.rdnut", "encoders.0.0.ffn.fn.1")             # GELU (exact/erf)
    emit_block_bundle("ffn_block.rdnut", "encoders.0.0.ffn",
                      ccm_tensors("encoders.0.0.ffn", ""), "CCM block: conv3x3->GELU->conv1x1")
    emit_block_bundle("dfm_block.rdnut", "encoders.0.0.dfm",
                      dfm_tensors("encoders.0.0.dfm", ""),
                      "DFM block: proj_in->chunk->mlp_h/w->gate/merge->proj_out")
    emit_block_bundle("encode_layer.rdnut", "encoders.0.0",
                      {**dfm_tensors("encoders.0.0.dfm", "dfm."),
                       **ccm_tensors("encoders.0.0.ffn", "ffn.")},
                      "EncodeLayer: x=dfm(x)+x; x=ffn(x)+x")
    emit_block_bundle("hfb_block.rdnut", "decoders.1.hfb",   # x & g both 64x64 -> g-resize is identity
                      hfb_tensors("decoders.1.hfb", ""),
                      "HFB block: conv_x/conv_g -> cross-GELU-gate -> concat -> proj_out")
    emit_block_bundle("ctr_block.rdnut", "decoders.1.ctr",   # frame 0: pre_h=None -> identity warp
                      ctr_tensors("decoders.1.ctr", ""),
                      "CTR block (frame0): to_q/to_kv -> cos-attn -> proj_out(cat[out,q2,depth])",
                      first=True)
    emit_block_bundle("upsample_block.rdnut", "ups.1",       # F.interpolate x2 (bilinear) + 3x3 conv
                      {"conv.weight": ok["ups.1.conv.weight"].numpy(),
                       "conv.bias":   ok["ups.1.conv.bias"].numpy()},
                      "Upsampling: interpolate x2 (bilinear) -> conv3x3")
    emit_block_bundle("decode_layer.rdnut", "decoders.1",    # frame 0 (CTR identity warp)
                      {**hfb_weights("decoders.1.hfb", "hfb."),
                       **ctr_weights("decoders.1.ctr", "ctr."),
                       **ccm_tensors("decoders.1.ffn", "ffn."),
                       "g":     golden_f_in1["decoders.1"][0],   # DecodeLayer input[1] (g-buffer)
                       "depth": golden_f_in2["decoders.1"][0]},  # DecodeLayer input[2] (depth)
                      "DecodeLayer (frame0): hfb(x,g)+x -> ctr(x,d)+x -> ffn(x)+x",
                      first=True)

    fc = golden.get("first_conv")
    print(f"\nForward OK. output {tuple(out.shape)}  "
          f"min {out.min():.4f} max {out.max():.4f} mean {out.mean():.4f}  "
          f"nan={bool(torch.isnan(out).any())}")
    print(f"captured {len(golden)} layer activations")
    if fc is not None:
        print(f"first_conv output {fc.shape}  mean {fc.mean():.5f}  std {fc.std():.5f}")
    print(f"saved -> {os.path.join(OUT_DIR, 'golden.npz')}")


if __name__ == "__main__":
    main()
