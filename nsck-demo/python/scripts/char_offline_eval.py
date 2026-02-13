"""Offline sanity checks for the NSCK character recognizer.

This script evaluates whether the char head produces non-uniform predictions
and whether different shapes produce different top-k outputs.

It uses the same core decode strategy as the server's `predict_char`:
- black-on-white canvas -> invert
- optional rotate/flip candidate selection (asis vs rot90+fliplr)
- analog feature decode using shared LIF membrane

Usage (PowerShell):
  python .\char_offline_eval.py --n 20

Notes:
- Requires torchvision EMNIST data already downloaded (or pass --download).
- Uses CPU by default for portability.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision import datasets

from python.core.neural.snn_qat import TaskAwareSNN, ternarize_weight


@dataclass
class DecodeResult:
    probs: torch.Tensor  # [C]
    orientation: str
    debug: Dict[str, float]


def _center_10x10(x: np.ndarray) -> np.ndarray:
    ys, xs = np.where(x > 0.1)
    if len(xs) == 0 or len(ys) == 0:
        return x
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = x[y0 : y1 + 1, x0 : x1 + 1]
    canvas = np.zeros((10, 10), dtype=np.float32)
    ch, cw = crop.shape
    if ch > 0 and cw > 0 and ch <= 10 and cw <= 10:
        oy = (10 - ch) // 2
        ox = (10 - cw) // 2
        canvas[oy : oy + ch, ox : ox + cw] = crop
        return canvas
    return x


def _decode_logits_membrane(model: TaskAwareSNN, grid_10: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    frames = np.stack([grid_10] * 4, axis=0)
    inp = torch.tensor(frames).unsqueeze(0).float().to(device)

    latent = model.encoder(inp)
    w_shared = ternarize_weight(model.fc_shared.weight)
    mem = model.lif_shared.init_leaky()
    cur = torch.nn.functional.linear(latent, w_shared, model.fc_shared.bias)
    spk, mem = model.lif_shared(cur, mem)

    head = model.heads["char_recognition"]
    w_actor = ternarize_weight(head["actor"].weight)

    out = torch.nn.functional.linear(mem, w_actor, head["actor"].bias)
    dbg = {
        "cur_mean": float(cur.detach().mean().cpu().item()),
        "mem_mean": float(mem.detach().mean().cpu().item()),
        "spk_mean": float(spk.detach().mean().cpu().item()),
        "mem_abs_mean": float(mem.detach().abs().mean().cpu().item()),
    }
    return out[0].detach().cpu(), dbg


def decode_canvas_like(model: TaskAwareSNN, img_gray: np.ndarray, device: torch.device) -> DecodeResult:
    img_10 = cv2.resize(img_gray, (10, 10), interpolation=cv2.INTER_AREA)
    img_float = img_10.astype(np.float32) / 255.0

    # black on white canvas -> invert
    img_float = 1.0 - img_float

    img_float = (img_float > 0.2).astype(np.float32) * img_float

    cand_a = _center_10x10(img_float)
    cand_b = _center_10x10(np.fliplr(np.rot90(img_float, k=1)))

    out_a, dbg_a = _decode_logits_membrane(model, cand_a, device)
    out_b, dbg_b = _decode_logits_membrane(model, cand_b, device)

    probs_a = torch.softmax(out_a, dim=0)
    probs_b = torch.softmax(out_b, dim=0)

    if float(probs_b.max()) > float(probs_a.max()):
        return DecodeResult(probs=probs_b, orientation="rot90+fliplr", debug=dbg_b)
    return DecodeResult(probs=probs_a, orientation="asis", debug=dbg_a)


def make_synthetic(symbol: str) -> np.ndarray:
    """Create a rough black-on-white 28x28 canvas-like image."""
    img = np.full((28, 28), 255, np.uint8)

    if symbol == "0":
        cv2.circle(img, (14, 14), 8, 0, 3)
    elif symbol == "1":
        cv2.line(img, (14, 6), (14, 22), 0, 3)
    elif symbol.lower() == "b":
        cv2.line(img, (10, 6), (10, 22), 0, 3)
        cv2.ellipse(img, (14, 12), (6, 5), 0, 270, 90, 0, 3)
        cv2.ellipse(img, (14, 18), (6, 5), 0, 270, 90, 0, 3)
    else:
        raise ValueError(f"Unknown synthetic symbol: {symbol}")

    return img


def topk(probs: torch.Tensor, k: int = 5) -> List[Tuple[int, float]]:
    vals, idx = torch.topk(probs, k=min(k, probs.numel()))
    return [(int(i.item()), float(v.item())) for v, i in zip(vals, idx)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="Number of random EMNIST samples")
    ap.add_argument("--download", action="store_true", help="Download EMNIST if missing")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "..", "snn_task_aware.pth"))
    args = ap.parse_args()

    device = torch.device("cpu")

    model = TaskAwareSNN(beta=0.5).to(device)
    model.register_task("char_recognition", 62)

    if os.path.exists(args.weights):
        sd = torch.load(args.weights, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
    model.eval()

    print("=== Synthetic canvas-like ===")
    for sym in ["0", "1", "b"]:
        img = make_synthetic(sym)
        res = decode_canvas_like(model, img, device)
        ent = float((-torch.clamp(res.probs, 1e-6, 1.0) * torch.log(torch.clamp(res.probs, 1e-6, 1.0))).sum().item())
        print(f"{sym}: pred={int(res.probs.argmax())} maxp={float(res.probs.max()):.4f} H={ent:.3f} ori={res.orientation} top5={topk(res.probs)} shared={res.debug}")

    print("\n=== EMNIST(ByClass) samples (decoded as if from canvas) ===")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_mnist"))
    ds = datasets.EMNIST(root=root, split="byclass", train=True, download=args.download)

    # deterministic subset
    rng = np.random.RandomState(0)
    idxs = rng.choice(len(ds), size=args.n, replace=False)

    maxps: List[float] = []
    ents: List[float] = []
    for i, idx in enumerate(idxs):
        img_pil, label = ds[idx]
        img = np.array(img_pil)
        res = decode_canvas_like(model, img, device)
        ent = float((-torch.clamp(res.probs, 1e-6, 1.0) * torch.log(torch.clamp(res.probs, 1e-6, 1.0))).sum().item())
        maxp = float(res.probs.max().item())
        maxps.append(maxp)
        ents.append(ent)
        print(f"{i:02d}: true={label:02d} pred={int(res.probs.argmax()):02d} maxp={maxp:.4f} H={ent:.3f} ori={res.orientation} top5={topk(res.probs)}")

    print("\nSummary:")
    print(f"  avg maxp = {sum(maxps)/len(maxps):.4f}")
    print(f"  avg H    = {sum(ents)/len(ents):.3f}")


if __name__ == "__main__":
    main()
