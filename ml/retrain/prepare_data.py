#!/usr/bin/env python3
"""Training data for NSS retraining: convert captures, assemble a set, check it.

    prepare_data.py convert  <capture root> <out dir> [--crop 256] [--threads N]
    prepare_data.py assemble <out dir> <source dir>... [--val-fraction 0.05] [--test-fraction 0]
    prepare_data.py check    <dir>... [--scale 2]

convert   Captures from Arm's Unreal plugin (EXR + JSON, model-gym/docs/nss/nss_dataset_specification.md)
          become full-frame safetensors in <out dir>/full and 256 px training crops in <out dir>/crops,
          through the model gym's safetensors writer.
assemble  <out dir>/{train,val,test} as symlinks to every sequence under the sources (crops from
          `convert`, Arm's Bistro splits), whole captures kept on one side of the split. A source that
          already has train/, val/ or test/ folders keeps its own split.
check     Every sequence: the keys, shapes and ranges the gym and the runtime rely on, one line each.
"""
import argparse
import hashlib
import os
import pathlib
import subprocess
import sys

import numpy as np
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[2]
GYM = ROOT / "ml" / "model-gym"


def sequences(path):
    return sorted(p for p in pathlib.Path(path).rglob("*.safetensors") if p.is_file())


def gym(args, cwd=GYM):
    cmd = [sys.executable, "-m", "scripts.safetensors_generator.safetensors_writer"] + args
    print("+", " ".join(str(a) for a in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def convert(args):
    out = pathlib.Path(args.out).resolve()
    full, crops = out / "full", out / "crops"
    gym(["-src", str(pathlib.Path(args.capture).resolve()), "-dst", str(full), "-reader", "NSSv1_0_1", "-extension", "exr",
         "-threads", str(args.threads), "-logging_output_dir", str(out / "logs")])
    gym(["-src", str(full), "-dst", str(crops), "-reader", "cropper", "-extension", "safetensors", "-crop_size", str(args.crop),
         "-threads", str(args.threads), "-logging_output_dir", str(out / "logs")])
    print("%d full sequences, %d crops" % (len(sequences(full)), len(sequences(crops))))


def assemble(args):
    out = pathlib.Path(args.out).resolve()
    counts = {"train": 0, "val": 0, "test": 0}
    for src in args.sources:
        src = pathlib.Path(src).resolve()
        presplit = [s for s in ("train", "val", "test") if (src / s).is_dir()]
        for seq in sequences(src):
            rel = seq.relative_to(src)
            if presplit:
                split = rel.parts[0] if rel.parts[0] in presplit else "train"
            else:
                # crops of one capture share its name: keep them together
                h = int(hashlib.sha1(seq.stem.encode()).hexdigest(), 16) % 10000 / 10000
                split = "val" if h < args.val_fraction else "test" if h < args.val_fraction + args.test_fraction else "train"
            link = out / split / src.name / rel
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(seq)
            counts[split] += 1
    for split, n in counts.items():
        print("%-5s %6d sequences  %s" % (split, n, out / split))


def check(args):
    bad = 0
    for path in args.dirs:
        for seq in sequences(path):
            problems = []
            try:
                f = safe_open(str(seq), "np")
            except Exception as e:  # noqa: BLE001
                print("%-60s unreadable (%s); an LFS pointer?" % (seq, str(e).splitlines()[0][:60]))
                bad += 1
                continue
            with f:
                keys = set(f.keys())
                meta = f.metadata() or {}
                for k in ("colour_linear", "ground_truth_linear", "depth", "motion", "motion_lr", "jitter", "exposure", "zNear",
                          "zFar", "FovY", "infinite_zFar", "render_size", "outDims", "viewProj"):
                    if k not in keys:
                        problems.append("no " + k)
                if problems:
                    print("%-60s %s" % (seq, ", ".join(problems)))
                    bad += 1
                    continue
                col = f.get_slice("colour_linear").get_shape()
                gt = f.get_slice("ground_truth_linear").get_shape()
                mv = f.get_slice("motion").get_shape()
                t = col[0]
                ratio = gt[2] / col[2]
                if abs(ratio - args.scale) > 1e-6 or abs(gt[3] / col[3] - args.scale) > 1e-6:
                    problems.append("ratio %.3g" % ratio)
                if tuple(mv[2:]) != tuple(gt[2:]) or tuple(f.get_slice("motion_lr").get_shape()[2:]) != tuple(col[2:]):
                    problems.append("motion size")
                if int(meta.get("Length", t)) != t:
                    problems.append("Length %s of %d frames" % (meta.get("Length"), t))
                jit = f.get_tensor("jitter").reshape(t, 2)
                if np.abs(jit).max() > 0.5 + 1e-4:
                    problems.append("jitter beyond half a pixel")
                if np.ptp(jit, axis=0).min() < 0.25:
                    problems.append("jitter barely moves")
                c = f.get_tensor("colour_linear").astype(np.float32)
                if not np.isfinite(c).all() or c.min() < 0:
                    problems.append("colour not finite or negative")
                d = f.get_tensor("depth").astype(np.float32)
                if d.min() < 0 or d.max() > 1:
                    problems.append("depth outside [0, 1]")
                m = f.get_tensor("motion_lr").astype(np.float32)
                if np.abs(m).max() > col[2]:
                    problems.append("motion larger than the frame")
                ex = f.get_tensor("exposure").reshape(-1)
                print("%-60s %3d frames %4dx%-4d -> %4dx%-4d  exposure %.2f..%.2f  |mv| %.2f px%s" %
                      (seq, t, col[3], col[2], gt[3], gt[2], ex.min(), ex.max(), np.abs(m).mean(),
                       ("  " + ", ".join(problems)) if problems else ""))
            bad += bool(problems)
    print("%d sequence(s) with problems" % bad)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("capture")
    c.add_argument("out")
    c.add_argument("--crop", type=int, default=256)
    c.add_argument("--threads", type=int, choices=range(1, 5), default=min(4, max(1, (os.cpu_count() or 2) // 2)),
                   help="the gym writer takes 1 to 4")
    a = sub.add_parser("assemble")
    a.add_argument("out")
    a.add_argument("sources", nargs="+")
    a.add_argument("--val-fraction", type=float, default=0.05)
    a.add_argument("--test-fraction", type=float, default=0.0)
    k = sub.add_parser("check")
    k.add_argument("dirs", nargs="+")
    k.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()
    return {"convert": convert, "assemble": assemble, "check": check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
