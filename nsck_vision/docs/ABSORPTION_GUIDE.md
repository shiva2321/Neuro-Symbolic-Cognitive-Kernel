# Model Absorption Guide

Step-by-step instructions for absorbing pretrained models into NSCK-UPMA.

## Prerequisites

```bash
pip install -e nsck_vision/
# For torchvision:
pip install torch torchvision
# For HuggingFace:
pip install transformers
```

---

## 1. Absorbing a torchvision Model (ResNet-50)

```python
from nsck_vision.system import NSCKVisionSystem
import torchvision.datasets as dsets
import torchvision.transforms as T

system = NSCKVisionSystem()

# Optional: provide a real dataset for richer absorption
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
dataset = dsets.ImageFolder("path/to/imagenet/val", transform=transform)
dataset_iter = ((img.numpy(), label) for img, label in dataset)

report = system.absorb(
    model_or_name="resnet50",
    domain="imagenet",
    dataset=dataset_iter,
    max_samples=1000,
    model_id="resnet50_imagenet",
)

print(f"Absorbed {report.n_concepts_absorbed} concepts")
print(f"Spearman ρ: {report.spearman_rho:.4f}")
print(f"HV/s: {report.hv_per_second:.0f}")
print(f"Rust backend: {report.rust_backend_active}")
```

**Expected output:**
```
Absorbed 1000 concepts
Spearman ρ: 0.4312
HV/s: 2450
Rust backend: True
```

---

## 2. Absorbing a HuggingFace Model (ViT)

```python
from nsck_vision.system import NSCKVisionSystem
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import numpy as np

system = NSCKVisionSystem()

# Load the HuggingFace ViT model
model_name = "google/vit-base-patch16-224"
processor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Create a feature-extraction callable
def vit_features(x):
    """Extract CLS token embedding from ViT."""
    import torch
    inputs = processor(images=Image.fromarray(x), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1][:, 0, :].numpy().squeeze()

# Absorb with your own dataset
dataset_iter = [
    (np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8), f"class_{i}")
    for i in range(100)
]

report = system.absorb(
    model_or_name=vit_features,
    domain="imagenet_vit",
    dataset=dataset_iter,
    max_samples=100,
    model_id="vit_base_imagenet",
)
print(f"ViT absorbed: {report.n_concepts_absorbed} concepts, ρ={report.spearman_rho:.3f}")
```

---

## 3. Absorbing Multiple Models in Parallel

Parallel absorption uses `ThreadPoolExecutor` internally and is safe for I/O-bound feature extraction:

```python
from nsck_vision.system import NSCKVisionSystem
import numpy as np

system = NSCKVisionSystem()
rng = np.random.default_rng(42)

def make_dataset(n, dim):
    return [(rng.standard_normal(dim).astype(np.float32), f"cls_{i%10}") for i in range(n)]

specs = [
    {
        "model": lambda x: x,   # identity (replace with real model)
        "model_id": "resnet18_medical",
        "domain": "medical",
        "dataset_iter": make_dataset(200, 512),
        "max_samples": 200,
    },
    {
        "model": lambda x: x,
        "model_id": "vit_satellite",
        "domain": "satellite",
        "dataset_iter": make_dataset(200, 768),
        "max_samples": 200,
    },
    {
        "model": lambda x: x,
        "model_id": "effnet_traffic",
        "domain": "traffic",
        "dataset_iter": make_dataset(200, 1280),
        "max_samples": 200,
    },
]

reports = system.absorb_parallel(specs)
for r in reports:
    print(f"{r.model_id}: {r.n_concepts_absorbed} concepts, ρ={r.spearman_rho:.3f}, passed={r.passed}")
```

---

## 4. Running Analysis with a Reference Model

Use `compare()` to get a side-by-side comparison between NSCK and a reference model:

```python
from nsck_vision.system import NSCKVisionSystem

system = NSCKVisionSystem()
# ... (absorb models first) ...

result = system.compare("test_image.jpg")

print(f"NSCK prediction:      {result.nsck_label} ({result.nsck_confidence:.1%})")
print(f"Reference prediction: {result.reference_label} ({result.reference_confidence:.1%})")
print(f"NSCK overrode ref:    {result.nsck_overrode_reference}")
print(f"Accuracy rating:      {result.accuracy_rating:.1%}")
print(f"Causal chain:         {result.causal_chain}")
```

---

## 5. Cross-Domain Knowledge Transfer

NSCK-UPMA supports querying absorbed knowledge across domains:

```python
from python.core.vision.absorption_memory import AbsorptionMemory
from python.core.vision.domain_tagger import DomainTagger
import python.core.vsa.hypervec_shim as hvs

mem = AbsorptionMemory()
tagger = DomainTagger()

# Register cross-domain similarity
tagger.set_cross_domain_weight("medical", "satellite", 0.2)
tagger.set_cross_domain_weight("medical", "traffic", 0.1)

# After absorbing models, query by domain
medical_records = mem.query_by_domain("medical")
print(f"Medical concepts: {[r.label for r in medical_records[:5]]}")

# Cross-domain similarity
sim = tagger.get_cross_domain_similarity("medical", "satellite")
print(f"Medical ↔ Satellite similarity: {sim:.3f}")

# Coverage stats
stats = tagger.domain_coverage_stats()
print(f"Total domains: {stats['n_domains']}, models: {stats['n_models']}")
```

---

## Tips

- **Use `strategy="svd_factored"`** (default) for best similarity preservation.
- **Use `strategy="random"`** for faster absorption when speed matters more than ρ.
- **Increase `max_samples`** (up to ~10,000) for better concept coverage.
- **Save/load state** with `system.save_state("path/")` and `system.load_state("path/")`.
