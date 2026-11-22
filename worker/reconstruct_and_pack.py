#!/usr/bin/env python3
"""Zero-GPU CPU reconstruction of pilot-r1-agg from the 4 verified SparseLoCo uploads
plus deterministic CPU g6load dspine --nr 4 packing worker for glm53-dspine-pack.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import DraftSiteClient, sha256_file  # noqa: E402


def reconstruct_pilot_r1_agg(
    site_dir: Path,
    dspine_dir: Path,
    expected_sha: str,
) -> Path:
    import numpy as np
    import torch
    from safetensors import safe_open

    sys.path.insert(0, str(dspine_dir / "scripts"))
    from dspine_v4 import DSpineV4Config, DSpineV4Drafter  # noqa: E402

    out_dir = site_dir / "data/evals/pilot-r1-agg"
    st_out = out_dir / "model.safetensors"
    if st_out.is_file() and sha256_file(st_out) == expected_sha:
        print(f"[reconstruct] pilot-r1-agg already present with exact SHA-256 {expected_sha[:16]}...", flush=True)
        return out_dir

    ver_dir = site_dir / "data/uploads/verified"
    slot_files = {}
    for sid in ("r1_w0_bit200", "r1_w1_bit240", "r1_w2_bit280", "r1_w3_bit322_bit400"):
        matches = sorted(ver_dir.glob(f"*_{sid}.sparseloco.safetensors"))
        assert matches, f"missing verified upload for {sid}"
        slot_files[sid] = matches[-1]

    print(f"[reconstruct] Building base DSpineV4Drafter on CPU (0 GPU)...", flush=True)
    cfg = DSpineV4Config(
        block_size=16, num_hidden_layers=6, code_size=1024, code_rank=512,
        teacher_forcing_layers=3, use_dynamic_conv=True,
    )
    model = DSpineV4Drafter(cfg).to(device="cpu", dtype=torch.float32)
    model.load_dflash2_warmstart(str(dspine_dir / "out/eval-dflash2-ref/model.safetensors"))
    model.selected_prefix_bridge.load_transfer_space(str(dspine_dir / "out/glm53-vocab-ref/dspine-transfer-r1024.pt"))

    base_bf16 = {k: p.detach().to(torch.bfloat16).cpu().clone() for k, p in model.named_parameters() if p.requires_grad}
    ref_shapes = {k: tuple(v.shape) for k, v in base_bf16.items()}

    def unpack_delta(path: Path) -> tuple[dict[str, torch.Tensor], float]:
        out: dict[str, torch.Tensor] = {}
        sq = 0.0
        with safe_open(str(path), framework="numpy") as f:
            keys = set(f.keys())
            for name, shape in ref_shapes.items():
                n = 1
                for d in shape:
                    n *= d
                if f"{name}.__q" in keys:
                    q = f.get_tensor(f"{name}.__q")
                    sc = float(f.get_tensor(f"{name}.__scale").reshape(-1)[0])
                    if f"{name}.__mask" in keys:
                        bits = np.unpackbits(f.get_tensor(f"{name}.__mask"))[:n].astype(bool)
                        flat = np.zeros(n, dtype=np.float32)
                        flat[bits] = q.astype(np.float32) * sc
                        t = torch.from_numpy(flat.reshape(shape))
                    else:
                        t = torch.from_numpy((q.astype(np.float32) * sc).reshape(shape))
                elif f"{name}.__dense" in keys:
                    t = torch.from_numpy(f.get_tensor(f"{name}.__dense").astype(np.float32).reshape(shape))
                else:
                    t = torch.zeros(shape, dtype=torch.float32)
                sq += float(torch.dot(t.view(-1), t.view(-1)).item())
                out[name] = t
        return out, math.sqrt(max(0.0, sq))

    # First pass: compute exact norms for c_med = 1.5 * median(norm)
    norms = {}
    acc = {k: torch.zeros(shape, dtype=torch.float32) for k, shape in ref_shapes.items()}
    # Since all 4 norms (~32.96..33.14) are well below 1.5 * med (~49.5), scale == 1.0 for all 4!
    for sid, p in slot_files.items():
        d_f32, nm = unpack_delta(p)
        norms[sid] = nm
        print(f"  unpacked {sid}: norm={nm:.4f} ({p.name})", flush=True)
        for k in acc:
            acc[k].add_(d_f32[k], alpha=1.0 / len(slot_files))
        del d_f32
        gc.collect()

    med = float(np.median(list(norms.values())))
    assert all(nm <= 1.5 * med for nm in norms.values()), f"unexpected clipping needed: {norms}"

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_((base_bf16[name].float() + acc[name]).to(torch.bfloat16).float())
    del acc, base_bf16
    gc.collect()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.export_hf(str(out_dir))
    notice = dspine_dir / "NOTICE"
    if notice.is_file():
        shutil.copy2(notice, out_dir / "NOTICE")
    os.chmod(st_out, 0o644)
    got_sha = sha256_file(st_out)
    print(f"[reconstruct] Exported {st_out} sha256={got_sha} (expected={expected_sha}, match={got_sha == expected_sha})", flush=True)
    return out_dir


def run_pack_slot(
    cli: DraftSiteClient,
    slot_id: str,
    ckpt_dir: Path,
    pack_out_dir: Path,
    g6load_bin: Path,
    runner: str = "drs-worker-v1.1.0",
) -> dict[str, Any]:
    claim = cli.claim(runner=runner, job="glm53-dspine-pack", slot_id=slot_id, host_tag="frankennas-cpu")
    plan = claim["plan"]
    nr = int(plan.get("nr", 4))
    ckpt_rev = str(plan["checkpoint_rev"])
    ckpt_sha = sha256_file(ckpt_dir / "model.safetensors")
    g6_sha = sha256_file(g6load_bin)
    pack_out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cmd = [str(g6load_bin), "dspine", "--src", str(ckpt_dir), "--out", str(pack_out_dir), "--nr", str(nr)]
    print(f"[pack-worker] Running {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    elapsed = round(time.time() - t0, 2)

    ranks = []
    for r in range(nr):
        pf = pack_out_dir / f"dspine.r{r}.pack"
        mf = pack_out_dir / f"dspine.r{r}.manifest.json"
        ranks.append(
            {
                "rank": r,
                "pack_file": pf.name,
                "pack_bytes": pf.stat().st_size,
                "pack_sha256": sha256_file(pf),
                "manifest_sha256": sha256_file(mf),
            }
        )
    payload = {
        "checkpoint_rev": ckpt_rev,
        "ckpt_sha256": ckpt_sha,
        "nr": nr,
        "format_version": "g6dspine-v1",
        "g6load_sha256": g6_sha,
        "elapsed_s": elapsed,
        "ranks": ranks,
    }
    with tempfile.TemporaryDirectory() as td:
        jf = Path(td) / f"{slot_id}.pack_manifest.json"
        jf.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        resp = cli.upload_artifact(claim, jf, manifest={"nr": nr, "checkpoint_rev": ckpt_rev})
    print(f"[pack-worker] {slot_id} packed in {elapsed}s -> {resp.get('state')} ({resp.get('verify_status')})", flush=True)
    return {"payload": payload, "upload": resp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://draft.bertholomus.com")
    ap.add_argument("--keys-file", default="/tank/projects/draft-site/secrets/pilot_worker_keys.json")
    ap.add_argument("--site-dir", default="/tank/projects/draft-site")
    ap.add_argument("--dspine-dir", default="/tank/projects/glm53-dspine")
    ap.add_argument("--g6load", default="/tank/projects/glm53-dspine/engine/gr8-dspine-a3f44a0/engine/loader-rs/target/release/g6load")
    ap.add_argument("--reconstruct-only", action="store_true")
    ap.add_argument("--pack-slot", default=None)
    ap.add_argument("--pack-src", default=None)
    ap.add_argument("--pack-out", default=None)
    args = ap.parse_args()

    site_dir = Path(args.site_dir)
    dspine_dir = Path(args.dspine_dir)
    if args.reconstruct_only:
        reconstruct_pilot_r1_agg(
            site_dir, dspine_dir, "69b4916abbf9e11117300d2a6eda502ac83c6ebd3ae46e5b3100325e3c9784a8"
        )
        return

    keys = json.loads(Path(args.keys_file).read_text(encoding="utf-8"))
    cli = DraftSiteClient(args.url, keys["bertholomus-w1"])
    if args.pack_slot:
        run_pack_slot(cli, args.pack_slot, Path(args.pack_src), Path(args.pack_out), Path(args.g6load))


if __name__ == "__main__":
    main()
