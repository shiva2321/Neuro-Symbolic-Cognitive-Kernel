"""
NSCK V18 — Cognitive Vocabulary
================================
A comprehensive semantic vocabulary used by SemanticBootstrapper to build a
semantically meaningful DistributionalCodebook.

Covers 9 domains with ≥ 40 words each (500+ total).
"""
from typing import List

COGNITIVE_VOCABULARY: List[str] = [
    # ── Cognition & Mind ────────────────────────────────────────────────────
    "brain", "memory", "learning", "thinking", "reasoning", "knowledge",
    "perception", "attention", "consciousness", "emotion", "feeling",
    "intelligence", "cognition", "neural", "thought", "mind", "understand",
    "believe", "imagine", "remember", "forget", "recall", "recognize",
    "awareness", "intention", "motivation", "desire", "goal", "decision",
    "judgment", "inference", "concept", "abstraction", "representation",
    "schema", "prototype", "category", "classification", "analogy",
    "creativity", "insight", "intuition", "logic", "deduction", "induction",
    "hypothesis", "theory", "model", "prediction", "expectation", "surprise",
    "curiosity", "focus", "distraction", "confusion", "clarity", "doubt",
    "certainty", "belief", "opinion", "wisdom",

    # ── Biology & Life ──────────────────────────────────────────────────────
    "cell", "gene", "evolve", "organism", "species", "DNA", "protein",
    "nucleus", "membrane", "chromosome", "genome", "mutation", "evolution",
    "natural", "selection", "adaptation", "ecosystem", "biodiversity",
    "metabolism", "respiration", "photosynthesis", "reproduction", "growth",
    "development", "tissue", "organ", "blood", "heart", "lung", "liver",
    "kidney", "muscle", "bone", "nerve", "neuron", "synapse", "hormone",
    "immune", "bacteria", "virus", "fungus", "plant", "animal", "mammal",
    "bird", "fish", "insect", "tree", "flower", "seed", "root", "leaf",
    "ecology", "habitat", "predator", "prey", "food", "chain", "web",

    # ── Physics & Math ──────────────────────────────────────────────────────
    "force", "energy", "mass", "gravity", "quantum", "algebra", "compute",
    "velocity", "acceleration", "momentum", "pressure", "temperature",
    "heat", "light", "wave", "particle", "field", "charge", "electron",
    "proton", "neutron", "atom", "molecule", "matter", "space", "time",
    "relativity", "entropy", "thermodynamics", "electromagnetism",
    "frequency", "wavelength", "amplitude", "resonance", "vibration",
    "vector", "scalar", "matrix", "tensor", "equation", "formula",
    "calculus", "geometry", "topology", "probability", "statistics",
    "number", "function", "derivative", "integral", "limit", "infinity",
    "symmetry", "dimension", "coordinate", "graph", "set", "proof",

    # ── Computing & AI ──────────────────────────────────────────────────────
    "code", "algorithm", "data", "model", "train", "predict",
    "network", "system", "program", "software", "hardware", "computer",
    "database", "query", "index", "search", "sort", "optimize",
    "machine", "learning", "deep", "neural", "layer", "weight",
    "gradient", "backpropagation", "loss", "accuracy", "precision",
    "recall", "classification", "regression", "clustering", "embedding",
    "vector", "dimension", "feature", "input", "output", "function",
    "compiler", "interpreter", "runtime", "memory", "process", "thread",
    "parallel", "distributed", "cloud", "server", "client", "protocol",
    "encryption", "security", "authentication", "interface", "API",

    # ── Actions & Verbs ──────────────────────────────────────────────────────
    "run", "build", "create", "destroy", "move", "connect", "separate",
    "learn", "teach", "explain", "describe", "analyze", "synthesize",
    "combine", "divide", "expand", "reduce", "increase", "decrease",
    "start", "stop", "continue", "change", "transform", "convert",
    "send", "receive", "store", "retrieve", "process", "compute",
    "measure", "estimate", "predict", "verify", "test", "evaluate",
    "design", "implement", "deploy", "monitor", "control", "regulate",
    "grow", "develop", "evolve", "adapt", "improve", "optimize",
    "find", "discover", "explore", "observe", "detect", "identify",
    "define", "classify", "organize", "group", "relate", "compare",

    # ── Entities & Objects ───────────────────────────────────────────────────
    "animal", "machine", "system", "structure", "process", "pattern",
    "object", "entity", "thing", "substance", "material", "element",
    "component", "part", "whole", "unit", "module", "block", "layer",
    "network", "graph", "tree", "path", "node", "edge", "link",
    "tool", "device", "instrument", "sensor", "motor", "engine",
    "vehicle", "building", "city", "country", "world", "universe",
    "resource", "artifact", "product", "result", "output", "input",
    "signal", "message", "symbol", "token", "word", "sentence", "text",
    "image", "sound", "video", "document", "file", "record", "data",
    "information", "knowledge", "fact", "rule", "law", "principle",

    # ── Emotions & Social ────────────────────────────────────────────────────
    "happy", "fear", "trust", "communicate", "cooperate", "conflict",
    "sad", "angry", "joy", "love", "hate", "surprise", "disgust",
    "anxiety", "stress", "calm", "excitement", "boredom", "interest",
    "satisfaction", "frustration", "pride", "shame", "guilt", "hope",
    "community", "society", "culture", "language", "communication",
    "relationship", "friendship", "family", "group", "team", "leader",
    "cooperation", "competition", "conflict", "negotiation", "agreement",
    "behavior", "attitude", "norm", "value", "ethics", "morality",
    "education", "government", "economy", "politics", "history", "art",
    "religion", "philosophy", "science", "technology", "medicine",

    # ── Temporal & Causal ────────────────────────────────────────────────────
    "before", "after", "cause", "effect", "sequence", "cycle", "change",
    "time", "past", "present", "future", "duration", "moment", "period",
    "begin", "end", "continue", "repeat", "delay", "speed", "slow",
    "fast", "early", "late", "now", "then", "always", "never", "often",
    "because", "therefore", "hence", "leads", "results", "produces",
    "enables", "prevents", "blocks", "triggers", "activates", "inhibits",
    "condition", "consequence", "implication", "dependency", "influence",
    "prior", "subsequent", "simultaneous", "concurrent", "sequential",
    "history", "origin", "evolution", "development", "progress", "decay",

    # ── Spatial & Geometric ──────────────────────────────────────────────────
    "near", "far", "inside", "outside", "above", "below", "connect",
    "left", "right", "front", "back", "top", "bottom", "center", "edge",
    "point", "line", "surface", "volume", "shape", "size", "distance",
    "position", "location", "direction", "orientation", "angle", "curve",
    "circle", "square", "triangle", "sphere", "cube", "plane", "space",
    "region", "boundary", "border", "area", "depth", "height", "width",
    "horizontal", "vertical", "diagonal", "parallel", "perpendicular",
    "adjacent", "opposite", "between", "within", "around", "along",
    "toward", "away", "through", "across", "over", "under", "beside",
]
