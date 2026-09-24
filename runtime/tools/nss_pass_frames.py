#!/usr/bin/env python3
"""Cut a frame sequence for the NSS pass test out of an Arm neural-graphics-dataset sequence.

The dataset stores motion as (y, x) pixels pointing forward in time and jitter as (y, x);
the SDK passes read motion as (x, y) with reproj = uv + motion / size. Both are converted
here, so the test feeds the passes exactly what a game would through the SDK.

    python3 runtime/tools/nss_pass_frames.py <sequence.safetensors> [--frames 8] [--size 124x116]
        [--out runtime/tools/golden/nss_frames.rdnut]

Sequences: https://huggingface.co/datasets/Arm/neural-graphics-dataset (nss/val/bistro/...).
"""
import argparse
import pathlib
import struct

import numpy as np
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[2]


def write_rdnut(path, tensors):
    with open(path, "wb") as f:
        f.write(b"RDNT" + struct.pack("<II", 1, len(tensors)))
        for name, a in tensors.items():
            a = np.ascontiguousarray(a, dtype=np.float32)
            f.write(struct.pack("<I", len(name)) + name.encode())
            f.write(struct.pack("<I", a.ndim) + struct.pack("<%dI" % a.ndim, *a.shape))
            f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--size", default="124x116", help="render WxH cropped from the top-left")
    ap.add_argument("--out", default=str(ROOT / "runtime/tools/golden/nss_frames.rdnut"))
    args = ap.parse_args()

    w, h = map(int, args.size.split("x"))
    f = safe_open(args.sequence, "np")
    t = slice(args.start, args.start + args.frames)

    def img(name):
        return f.get_tensor(name)[t].astype(np.float32).transpose(0, 2, 3, 1)

    colour = img("colour_linear")[:, :h, :w]
    scale = int(round(float(f.get_tensor("scale")[0, 0])))
    truth = img("ground_truth_linear")[:, :h * scale, :w * scale]
    motion = img("motion_lr")[:, :h, :w]
    jitter = f.get_tensor("jitter")[t].reshape(-1, 2).astype(np.float32)
    ones = np.ones(colour.shape[:3] + (1,), np.float32)

    out = {
        "colour": np.concatenate([colour, ones], -1),
        "depth": img("depth")[:, :h, :w],
        "motion": -motion[..., ::-1],
        # display resolution motion in display pixels, for FSR's DISPLAY_RESOLUTION_MOTION_VECTORS
        "motion_hr": -img("motion")[:, :h * scale, :w * scale][..., ::-1],
        "jitter": jitter[:, ::-1],
        "exposure": np.exp(f.get_tensor("exposure")[t].reshape(-1)),
        # near, far, vertical fov, infinite far plane: what a game passes to the SDK
        "camera": np.stack([f.get_tensor("zNear")[t].reshape(-1), f.get_tensor("zFar")[t].reshape(-1),
                            f.get_tensor("FovY")[t].reshape(-1),
                            f.get_tensor("infinite_zFar")[t].reshape(-1).astype(np.float32)], -1),
        "truth": np.concatenate([truth, np.ones(truth.shape[:3] + (1,), np.float32)], -1),
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_rdnut(args.out, out)
    print("%s: %d frames, render %dx%d, display %dx%d" % (args.out, colour.shape[0], w, h, w * scale, h * scale))


if __name__ == "__main__":
    main()
