#!/usr/bin/env python3
"""Retrain NSS on new data and put the result into the runtime.

    retrain.py <run dir> --data <dir with train/ val/ [test/]> [options]

Stages, in order, each resumable from its output (--stages picks a subset):
    config    the model gym configuration for this run, <run>/nss_config.json
    fp32      fine-tune the fp32 model from --from (the released weights by default)
    qat       quantisation-aware fine-tune of the fp32 result
    evaluate  the gym's metrics on the test split for the start, fp32 and QAT weights
    export    runtime/tools/nss_export.py: manifest, INT8 weights and goldens under <run>/export;
              --install copies them into runtime/models/NSS_INT8 and runtime/tools/golden and
              checks them with the built test_engine_core when there is one

Needs the model gym (pip install ml/model-gym, a CUDA GPU; --cpu runs the torch backend for smoke
tests) and data from prepare_data.py. See README.md for the knobs that matter.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "ml/model-gym/src/ng_model_gym/usecases/nss/configs/nss_v1_template.json"
RELEASED = ROOT / "ml/models/nss/nss_v1_0_1_high_fp32.pt"
STAGES = ("config", "fp32", "qat", "evaluate", "export")


class Run:
    def __init__(self, args):
        self.args = args
        self.dir = pathlib.Path(args.run).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config = self.dir / "nss_config.json"
        self.state_path = self.dir / "run.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.gym = args.gym or shutil.which("ng-model-gym")
        if not self.gym and any(s in args.stages for s in ("fp32", "qat", "evaluate")):
            sys.exit("ng-model-gym not found: pip install ml/model-gym, or pass --gym")

    def save(self, **kw):
        self.state.update(kw)
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n")

    def sh(self, cmd, cwd=None):
        cmd = [str(c) for c in cmd]
        print("+", " ".join(cmd), flush=True)
        t = time.time()
        subprocess.run(cmd, cwd=cwd, check=True)
        print("  %.0f s" % (time.time() - t), flush=True)

    def latest_dir(self, kind):
        # the gym puts each training start in a timestamped folder
        runs = sorted(p for p in (self.dir / kind).glob("*") if p.is_dir() and any(p.glob("ckpt-*.pt")))
        if not runs:
            sys.exit("no checkpoints under %s" % (self.dir / kind))
        return runs[-1]

    def checkpoint(self, kind):
        d = self.latest_dir(kind)
        best = d / "best-validated-ckpt.pt"
        if best.exists():
            return best
        return max(d.glob("ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[1]))

    def start_weights(self):
        p = RELEASED if self.args.start == "released" else pathlib.Path(self.args.start).resolve()
        if not p.exists() or p.stat().st_size < 4096:
            sys.exit("%s is missing or an LFS pointer (git -C ml/models/nss lfs pull)" % p)
        return p

    # --------------------------------------------------------------------------------- stages

    def stage_config(self):
        a = self.args
        data = pathlib.Path(a.data).resolve()
        cfg = json.loads(TEMPLATE.read_text())
        cfg["model"].update({"scale": float(a.scale), "recurrent_samples": a.unroll, "quality": "high",
                             "processing_backend": "torch" if a.cpu else "slang"})
        cfg["dataset"]["path"] = {"train": str(data / "train"), "validation": str(data / "val"),
                                  "test": str(data / "test") if (data / "test").is_dir() else None}
        cfg["dataset"]["num_workers"] = 2 if a.cpu else 4
        cfg["output"]["dir"] = str(self.dir / "output")
        cfg["output"]["tensorboard_output_dir"] = str(self.dir / "tensorboard")
        cfg["output"]["export"]["vgf_output_dir"] = str(self.dir / "vgf")
        t = cfg["train"]
        t["batch_size"] = a.batch
        t["compile"] = not a.cpu
        t["perform_validate"] = (data / "val").is_dir()
        t["fp32"]["number_of_epochs"] = a.epochs
        t["fp32"]["checkpoints"]["dir"] = str(self.dir / "fp32")
        t["fp32"]["optimizer"]["learning_rate"] = a.lr
        t["qat"]["number_of_epochs"] = a.qat_epochs
        t["qat"]["checkpoints"]["dir"] = str(self.dir / "qat")
        t["qat"]["optimizer"]["learning_rate"] = a.qat_lr
        if a.loss_temporal is not None:
            t["loss_args"]["temporal_reg_weight"] = a.loss_temporal
        self.config.write_text(json.dumps(cfg, indent=4) + "\n")
        self.save(config=str(self.config), data=str(data), has_test=cfg["dataset"]["path"]["test"] is not None)
        print("wrote", self.config)

    def stage_fp32(self):
        cmd = [self.gym, "-c", self.config, "train", "--no-evaluate"]
        cmd += ["--resume", self.latest_dir("fp32")] if self.args.resume else ["--finetune", self.start_weights()]
        self.sh(cmd)
        self.save(fp32=str(self.checkpoint("fp32")))
        print("fp32 checkpoint", self.state["fp32"])

    def stage_qat(self):
        cmd = [self.gym, "-c", self.config, "qat", "--no-evaluate"]
        cmd += ["--resume", self.latest_dir("qat")] if self.args.resume else ["--finetune", self.state.get("fp32") or self.checkpoint("fp32")]
        self.sh(cmd)
        self.save(qat=str(self.checkpoint("qat")))
        print("qat checkpoint", self.state["qat"])

    def stage_evaluate(self):
        if not self.state.get("has_test"):
            print("no test split, evaluation skipped")
            return
        out = self.dir / "output"
        results = {}
        models = [("start", self.start_weights(), "fp32")]
        for kind, mode in (("fp32", "fp32"), ("qat", "qat_int8")):
            if (self.dir / kind).is_dir():
                self.state.setdefault(kind, str(self.checkpoint(kind)))
                models.append((kind, self.state[kind], mode))
        for name, path, kind in models:
            before = set(out.glob("eval_metrics_*.json"))
            self.sh([self.gym, "-c", self.config, "evaluate", "--model-path", path, "--model-type", kind])
            new = set(out.glob("eval_metrics_*.json")) - before
            if new:
                results[name] = json.loads(max(new).read_text())
        # the gym streams each metric per frame; the last frame holds the run's mean
        summary = {name: {m: v[max(v, key=int)] for m, v in r.items() if isinstance(v, dict) and v} for name, r in results.items()}
        self.save(evaluate=summary)
        metrics = sorted({m for r in summary.values() for m in r})
        print("%-6s" % "" + "".join("%18s" % m for m in metrics))
        for name, r in summary.items():
            print("%-6s" % name + "".join("%18.4f" % r[m] if isinstance(r.get(m), (int, float)) else "%18s" % "-" for m in metrics))

    def stage_export(self):
        fp32 = self.state.get("fp32") or self.checkpoint("fp32")
        qat = self.state.get("qat") or self.checkpoint("qat")
        export = self.dir / "export"
        golden, model = export / "golden", export / "model"
        self.sh([sys.executable, ROOT / "runtime/tools/nss_export.py", "--fp32", fp32, "--int8", qat, "--out", golden, "--models", model])
        for name in ("NOTICE.md", "third_party_licenses_and_copyright_notices.txt", "Arm_AI_Model_Community_License_v1_0_PRE-1154.pdf"):
            src = ROOT / "runtime/models/NSS_INT8" / name
            if src.exists():
                shutil.copy2(src, model / name)
        self.save(export=str(export))
        print("exported to", export)
        if not self.args.install:
            print("--install puts it into runtime/models/NSS_INT8 and runtime/tools/golden")
            return
        for src, dst in ((model, ROOT / "runtime/models/NSS_INT8"), (golden, ROOT / "runtime/tools/golden")):
            for f in src.iterdir():
                shutil.copy2(f, dst / f.name)
        print("installed; rebuild the runtime to embed the new weights")
        tests = [ROOT / "build/linux/runtime/test_engine_core", ROOT / "build/windows/runtime/Release/test_engine_core.exe"]
        for t in tests:
            if t.exists():
                self.sh([t, ROOT / "runtime/models/NSS_INT8", ROOT / "runtime/tools/golden"])
                break
        else:
            print("no build found: run test_engine_core runtime/models/NSS_INT8 runtime/tools/golden after building")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory (created)")
    ap.add_argument("--data", help="prepare_data.py assemble output: train/, val/, optional test/")
    ap.add_argument("--stages", default=",".join(STAGES), help="comma separated subset of " + " ".join(STAGES))
    ap.add_argument("--from", dest="start", default="released", help="fp32 .pt to fine-tune from (default: the released weights)")
    ap.add_argument("--resume", action="store_true", help="continue the fp32 or qat stage from its checkpoints")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--qat-epochs", type=int, default=2)
    ap.add_argument("--unroll", type=int, default=16, help="recurrent frames per training sample")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--qat-lr", type=float, default=1e-4)
    ap.add_argument("--loss-temporal", type=float, default=None, help="temporal (flicker) loss weight, template 0.7")
    ap.add_argument("--cpu", action="store_true", help="torch processing backend, no compile: smoke tests only")
    ap.add_argument("--gym", help="ng-model-gym executable")
    ap.add_argument("--install", action="store_true", help="export: install into the runtime and verify")
    args = ap.parse_args()
    args.stages = [s for s in args.stages.split(",") if s]
    for s in args.stages:
        if s not in STAGES:
            sys.exit("unknown stage " + s)
    run = Run(args)
    if "config" in args.stages and not args.data:
        sys.exit("--data is required for the config stage")
    if "config" not in args.stages and not run.config.exists():
        sys.exit("no %s: run the config stage first" % run.config)
    for s in args.stages:
        print("== " + s, flush=True)
        getattr(run, "stage_" + s)()


if __name__ == "__main__":
    main()
