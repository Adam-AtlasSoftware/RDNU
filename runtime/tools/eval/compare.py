#!/usr/bin/env python3
"""Compare upscaler outputs from RDNU captures (RDNU_CAPTURE, see runtime/src/provider/rdnu_capture.h).

    compare.py <capture dir> [<capture dir B>] [--truth frames.rdnut] [--context N] [--exposure E]
               [--crop x,y,w,h ...] [--html report.html] [--min-psnr dB]

One directory: its "output" (RDNU) against its "reference" (AMD's upscaler, RDNU_COMPARE) when
captured. Two directories: the first's output against the second's. --truth adds the ground
truth ("truth", T x H x W x C, from runtime/tools/nss_pass_frames.py), matched by frame index.

Metrics follow Arm's model gym: Reinhard-tonemapped exposed colour in [0, 1], PSNR, SSIM
(11x11 Gaussian, sigma 1.5), tPSNR (PSNR of frame-to-frame differences); LPIPS (alex) and FLIP
when the lpips and flip_evaluator packages are installed.
"""
import argparse
import base64
import glob
import html
import os
import re
import struct
import zlib

import numpy as np


def read_rdnut(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"RDNT", path
    count = struct.unpack_from("<I", data, 8)[0]
    off, out = 12, {}
    for _ in range(count):
        n = struct.unpack_from("<I", data, off)[0]
        name = data[off + 4:off + 4 + n].decode()
        off += 4 + n
        nd = struct.unpack_from("<I", data, off)[0]
        dims = struct.unpack_from("<%dI" % nd, data, off + 4)
        off += 4 + 4 * nd
        size = int(np.prod(dims))
        out[name] = np.frombuffer(data, np.float32, size, off).reshape(dims)
        off += 4 * size
    return out


def load_capture(path, context):
    files = sorted(glob.glob(os.path.join(path, "c*_f*.rdnut")))
    by_context = {}
    for f in files:
        c, i = map(int, re.search(r"c(\d+)_f(\d+)", os.path.basename(f)).groups())
        by_context.setdefault(c, []).append((i, f))
    if not by_context:
        raise SystemExit("no captures in " + path)
    c = context if context is not None else max(by_context, key=lambda k: len(by_context[k]))
    return {i: read_rdnut(f) for i, f in by_context[c]}


def tonemap(rgb, exposure):
    x = np.maximum(rgb[..., :3] * exposure, 0)
    return x / (1 + x)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1 / mse) if mse > 0 else float("inf")


def blur(x, sigma=1.5, taps=11):
    k = np.exp(-0.5 * ((np.arange(taps) - taps // 2) / sigma) ** 2)
    k /= k.sum()
    for axis in (0, 1):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (taps // 2, taps // 2)
        p = np.pad(x, pad, mode="reflect")
        n = x.shape[axis]
        x = sum(w * np.take(p, np.arange(i, i + n), axis=axis) for i, w in enumerate(k))
    return x


def ssim(a, b):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s.mean())


class Perceptual:
    """LPIPS and FLIP when their packages are installed."""

    def __init__(self):
        self.lpips = self.flip = None
        try:
            import lpips
            import torch
            self.torch, self.lpips = torch, lpips.LPIPS(net="alex", verbose=False)
        except ImportError:
            pass
        try:
            import flip_evaluator
            self.flip = flip_evaluator
        except ImportError:
            pass

    def names(self):
        return (["LPIPS"] if self.lpips else []) + (["FLIP"] if self.flip else [])

    def __call__(self, a, b):
        out = []
        if self.lpips:
            t = lambda x: self.torch.from_numpy(x.transpose(2, 0, 1)[None] * 2 - 1).float()
            with self.torch.no_grad():
                out.append(float(self.lpips(t(a), t(b))))
        if self.flip:
            out.append(float(self.flip.evaluate(b, a, "LDR")[1]))
        return out


def png(rgb):
    """PNG bytes of an sRGB-encoded float image in [0, 1]."""
    img = (np.clip(rgb, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def worst_blocks(err, size, count):
    h, w = err.shape
    s = size
    blocks = err[:h // s * s, :w // s * s].reshape(h // s, s, w // s, s).mean(axis=(1, 3))
    picks = []
    for i in np.argsort(blocks, axis=None)[::-1]:
        y, x = divmod(int(i), blocks.shape[1])
        if all(abs(y - py) > 1 or abs(x - px) > 1 for py, px in picks):
            picks.append((y, x))
        if len(picks) == count:
            break
    return [(x * s, y * s, s, s) for y, x in picks]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("capture_b", nargs="?")
    ap.add_argument("--truth", help=".rdnut with a 'truth' tensor (nss_pass_frames.py)")
    ap.add_argument("--context", type=int, help="capture context id (default: the longest)")
    ap.add_argument("--exposure", type=float, help="exposure for tonemapping (default: from the capture)")
    ap.add_argument("--crop", action="append", default=[], help="x,y,w,h in output pixels")
    ap.add_argument("--html", help="write a report with crops")
    ap.add_argument("--min-psnr", type=float, help="fail unless RDNU's mean PSNR reaches this")
    args = ap.parse_args()

    a = load_capture(args.capture, args.context)
    names = ["RDNU"]
    if args.capture_b:
        b = load_capture(args.capture_b, args.context)
        frames = sorted(set(a) & set(b))
        images = {i: [a[i]["output"], b[i]["output"]] for i in frames}
        names.append(os.path.basename(os.path.normpath(args.capture_b)))
    else:
        frames = sorted(a)
        images = {i: [a[i]["output"]] + ([a[i]["reference"]] if "reference" in a[i] else []) for i in frames}
        if "reference" in a[frames[0]]:
            names.append("AMD")
    truth = read_rdnut(args.truth)["truth"] if args.truth else None

    def exposure(i):
        if args.exposure:
            return args.exposure
        f = a[i]
        e = float(f["exposure"].flat[0]) if "exposure" in f else 1.0
        return e / float(f["pre_exposure"][0] or 1)

    perceptual = Perceptual()
    rows, prev, tsum, crops_src = [], {}, {}, None
    for i in frames:
        h, w = int(a[i]["upscale"][1]), int(a[i]["upscale"][0])
        ex = exposure(i)
        outs = [tonemap(x[:h, :w], ex) for x in images[i]]
        ref = tonemap(truth[i, :h, :w], ex) if truth is not None and i < len(truth) else None
        if ref is None and len(outs) > 1:
            ref, outs, cand = outs[-1], outs[:-1], names[:-1]
        else:
            cand = names
        if ref is None:
            raise SystemExit("nothing to compare against: capture a reference (RDNU_COMPARE) or pass --truth")
        row = {"frame": i}
        for n, o in zip(cand, outs):
            row[n] = [psnr(o, ref), ssim(o, ref)] + perceptual(o, ref)
            if n in prev and prev[n][0].shape == o.shape:
                tsum.setdefault(n, []).append(psnr(o - prev[n][0], ref - prev[n][1]))
            prev[n] = (o, ref)
        rows.append(row)
        crops_src = (outs, ref, cand)

    metrics = ["PSNR", "SSIM"] + perceptual.names()
    against = "truth" if truth is not None else names[-1]
    cand = crops_src[2]
    print("%-8s" % "frame" + "".join("%12s %-7s" % (n, m) for n in cand for m in metrics))
    for r in rows:
        print("%-8d" % r["frame"] + "".join("%20.4f" % v for n in cand for v in r[n]))
    print("mean    " + "".join("%20.4f" % np.mean([r[n][k] for r in rows]) for n in cand for k in range(len(metrics))))
    for n in cand:
        if n in tsum:
            print("%s tPSNR against %s: %.2f dB" % (n, against, np.mean(tsum[n])))
    mean_psnr = float(np.mean([r[cand[0]][0] for r in rows]))

    if args.html:
        outs, ref, cand = crops_src
        err = np.abs(outs[0] - ref).mean(axis=2)
        boxes = [tuple(map(int, c.split(","))) for c in args.crop] or worst_blocks(err, max(16, min(err.shape) // 8), 4)
        img = lambda x: '<img src="data:image/png;base64,%s">' % base64.b64encode(png(x)).decode()
        zoom = lambda x: np.repeat(np.repeat(x, 4, axis=0), 4, axis=1)
        cells = "".join("<th>%s</th>" % html.escape(n) for n in cand + [against])
        crop_rows = "".join("<tr><td>%d,%d %dx%d</td>%s</tr>" % (x, y, w, h, "".join(
            "<td>%s</td>" % img(zoom(o[y:y + h, x:x + w])) for o in outs + [ref])) for x, y, w, h in boxes)
        table = "".join("<tr><td>%d</td>%s</tr>" % (r["frame"], "".join("<td>%.4f</td>" % v for n in cand for v in r[n]))
                        for r in rows)
        head = "".join("<th>%s %s</th>" % (html.escape(n), m) for n in cand for m in metrics)
        with open(args.html, "w") as f:
            f.write("""<!doctype html><meta charset="utf-8"><title>RDNU comparison</title>
<style>body{font:14px system-ui;margin:16px;background:#fff;color:#111}td,th{padding:2px 8px;text-align:right}
img{image-rendering:pixelated;max-width:100%%}</style>
<h1>RDNU comparison</h1><p>%s against %s, frames %d-%d.</p>
<h2>Last frame</h2><table><tr>%s</tr><tr>%s</tr></table>
<h2>Crops (4x)</h2><table><tr><th></th>%s</tr>%s</table>
<h2>Per frame</h2><table><tr><th>frame</th>%s</tr>%s</table>""" % (
                html.escape(", ".join(cand)), html.escape(against), frames[0], frames[-1], cells,
                "".join("<td>%s</td>" % img(o) for o in outs + [ref]), cells, crop_rows, head, table))
        print("report", args.html)
    if args.min_psnr is not None and mean_psnr < args.min_psnr:
        raise SystemExit("%s mean PSNR %.2f dB is below %.2f" % (cand[0], mean_psnr, args.min_psnr))


if __name__ == "__main__":
    main()
