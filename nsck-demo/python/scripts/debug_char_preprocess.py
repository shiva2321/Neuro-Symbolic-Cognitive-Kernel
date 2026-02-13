"""Offline smoke test for character preprocessing and model inference.

This script runs the same preprocessing used by the server on a few EMNIST
samples (if available) and prints top-5 predictions. It’s meant to quickly
answer: "are we still stuck predicting 0?" and "is the input polarity/orientation
reasonable?".

Usage (PowerShell):
  python .\python\debug_char_preprocess.py

Optional:
  python .\python\debug_char_preprocess.py --n 16
"""

from __future__ import annotations

import argparse

import numpy as np


def server_like_preprocess(img_28: np.ndarray) -> np.ndarray:
    """Approximate the `predict_char` preprocessing pipeline.

    Input:
      img_28: uint8 grayscale image (H,W)

    Output:
      img_10: float32 (10,10) in [0,1]
    """
    import cv2

    img_10 = cv2.resize(img_28, (10, 10), interpolation=cv2.INTER_AREA)
    img_float = img_10.astype(np.float32) / 255.0

    # polarity auto-fix
    if float(img_float.mean()) > 0.5:
        img_float = 1.0 - img_float

    # orientation fix
    img_float = np.rot90(img_float, k=1)
    img_float = np.fliplr(img_float)

    # threshold
    img_float = (img_float > 0.2).astype(np.float32) * img_float

    # center via bbox
    ys, xs = np.where(img_float > 0.1)
    if len(xs) and len(ys):
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        crop = img_float[y0 : y1 + 1, x0 : x1 + 1]
        canvas = np.zeros((10, 10), dtype=np.float32)
        ch, cw = crop.shape
        if ch > 0 and cw > 0 and ch <= 10 and cw <= 10:
            oy = (10 - ch) // 2
            ox = (10 - cw) // 2
            canvas[oy : oy + ch, ox : ox + cw] = crop
            img_float = canvas

    return img_float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of samples")
    args = ap.parse_args()

    import torch
    from torchvision import datasets

    from python.core.neural.snn_qat import TaskAwareSNN
    from python.utilities.concept_mapper import ConceptMapper

    model = TaskAwareSNN(beta=0.5)
    model.register_task("char_recognition", 62)

    # Load weights if present (same filename as server)
    import os

    model_path = os.path.join(os.path.dirname(__file__), "..", "snn_task_aware.pth")
    model_path = os.path.abspath(model_path)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights: {model_path}")
        except Exception as e:
            print(f"Could not load weights: {e}")
    else:
        print(f"No weights found at {model_path}; using random weights")

    model.eval()
    mapper = ConceptMapper()

    # Try EMNIST (byclass)
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_mnist"))
    try:
        ds = datasets.EMNIST(root=data_root, split="byclass", train=True, download=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load/download EMNIST into {data_root}. "
            f"If you’re offline, run the server once with EMNIST enabled or place the dataset there. Error: {e}"
        )

    idxs = np.random.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    for i, idx in enumerate(idxs):
        img_pil, label = ds[idx]
        img = np.array(img_pil)

        img_10 = server_like_preprocess(img)
        frames = np.stack([img_10] * 4, axis=0)
        inp = torch.tensor(frames).unsqueeze(0).float()

        with torch.no_grad():
            logits, _ = model(inp, task_name="char_recognition")
            probs = torch.softmax(logits, dim=1)[0]
            topv, topi = torch.topk(probs, k=5)

        topk = [(int(ii.item()), float(vv.item())) for vv, ii in zip(topv, topi)]
        pred = int(topi[0].item())
        print(
            f"#{i:02d} true={label:02d} pred={pred:02d} conf={float(topv[0]):.3f} "
            f"pred_label={mapper.get_explanation(pred)} top5={topk} img_mean={float(img_10.mean()):.3f}"
        )


if __name__ == "__main__":
    main()
