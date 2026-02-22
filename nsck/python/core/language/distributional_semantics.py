"""Distributional Semantics via VSA context windows for NSCK V3."""
from __future__ import annotations
import logging
import pickle
from typing import List, Dict, Optional
import python.core.vsa.hypervec_shim as hypervec_rs

logger = logging.getLogger(__name__)

HyperVector = hypervec_rs.HyperVector

BUILTIN_CORPUS: List[List[str]] = [
    # ── Basic world facts ──────────────────────────────────────────────────
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "in", "the", "park"],
    ["cats", "and", "dogs", "are", "common", "pets"],
    ["the", "sun", "shines", "in", "the", "sky"],
    ["water", "flows", "in", "rivers", "and", "oceans"],
    ["plants", "need", "water", "and", "sunlight", "to", "grow"],
    ["animals", "eat", "food", "to", "survive"],
    ["humans", "are", "social", "animals"],
    ["science", "studies", "the", "natural", "world"],
    ["physics", "and", "chemistry", "are", "natural", "sciences"],
    ["biology", "studies", "living", "organisms"],
    ["mathematics", "is", "the", "language", "of", "science"],
    ["computers", "process", "data", "and", "information"],
    ["the", "internet", "connects", "people", "worldwide"],
    ["cities", "are", "large", "human", "settlements"],
    ["forests", "provide", "oxygen", "and", "shelter"],
    ["the", "earth", "orbits", "the", "sun"],
    ["the", "moon", "orbits", "the", "earth"],
    ["gravity", "pulls", "objects", "toward", "the", "earth"],
    ["light", "travels", "faster", "than", "sound"],
    ["energy", "cannot", "be", "created", "or", "destroyed"],
    ["atoms", "make", "up", "all", "matter"],
    ["cells", "are", "the", "basic", "unit", "of", "life"],
    ["genes", "carry", "genetic", "information"],
    ["evolution", "explains", "the", "diversity", "of", "life"],
    ["the", "brain", "controls", "the", "nervous", "system"],
    ["language", "enables", "communication", "between", "humans"],
    ["knowledge", "is", "power", "and", "freedom"],
    ["technology", "changes", "human", "society"],
    ["art", "expresses", "human", "emotion", "and", "culture"],
    ["music", "and", "art", "are", "creative", "activities"],
    ["history", "records", "past", "human", "events"],
    ["the", "climate", "affects", "living", "conditions"],
    ["food", "provides", "energy", "for", "living", "organisms"],
    ["water", "is", "essential", "for", "life"],
    ["oxygen", "is", "needed", "for", "breathing"],
    ["the", "heart", "pumps", "blood", "through", "the", "body"],
    ["the", "lungs", "exchange", "oxygen", "and", "carbon", "dioxide"],
    ["trees", "convert", "carbon", "dioxide", "into", "oxygen"],
    ["rivers", "flow", "from", "mountains", "to", "the", "sea"],
    ["clouds", "form", "from", "water", "vapor", "in", "the", "atmosphere"],
    ["rain", "falls", "from", "clouds", "to", "the", "ground"],
    ["the", "ocean", "covers", "most", "of", "the", "earth"],
    ["birds", "can", "fly", "because", "they", "have", "wings"],
    ["fish", "live", "and", "breathe", "in", "water"],
    ["the", "cat", "chased", "the", "mouse", "across", "the", "room"],
    ["students", "learn", "new", "skills", "at", "school"],
    ["teachers", "help", "students", "understand", "concepts"],
    ["doctors", "treat", "patients", "with", "medicine"],
    ["engineers", "design", "and", "build", "structures"],
    # ── Physics / thermodynamics ───────────────────────────────────────────
    ["heat", "flows", "from", "hot", "to", "cold", "objects"],
    ["temperature", "measures", "the", "average", "kinetic", "energy", "of", "particles"],
    ["entropy", "tends", "to", "increase", "in", "isolated", "systems"],
    ["force", "equals", "mass", "times", "acceleration"],
    ["momentum", "is", "conserved", "in", "collisions"],
    ["electricity", "flows", "through", "conductors"],
    ["magnetism", "and", "electricity", "are", "related", "forces"],
    ["waves", "carry", "energy", "through", "space"],
    ["pressure", "increases", "with", "depth", "in", "fluids"],
    ["nuclear", "reactions", "release", "enormous", "amounts", "of", "energy"],
    ["quantum", "mechanics", "describes", "particles", "at", "very", "small", "scales"],
    ["relativity", "shows", "that", "space", "and", "time", "are", "connected"],
    # ── Chemistry / matter ────────────────────────────────────────────────
    ["molecules", "are", "made", "of", "atoms", "bonded", "together"],
    ["chemical", "reactions", "break", "and", "form", "bonds"],
    ["acids", "release", "hydrogen", "ions", "in", "solution"],
    ["bases", "accept", "hydrogen", "ions", "in", "solution"],
    ["catalysts", "speed", "up", "chemical", "reactions"],
    ["proteins", "are", "chains", "of", "amino", "acids"],
    ["DNA", "carries", "the", "genetic", "code", "of", "life"],
    ["enzymes", "are", "biological", "catalysts"],
    ["carbon", "is", "the", "backbone", "of", "organic", "molecules"],
    ["water", "is", "a", "polar", "molecule", "with", "unique", "properties"],
    ["oxidation", "involves", "loss", "of", "electrons"],
    ["reduction", "involves", "gain", "of", "electrons"],
    # ── Biology / life sciences ────────────────────────────────────────────
    ["the", "nervous", "system", "sends", "signals", "through", "neurons"],
    ["neurons", "communicate", "via", "electrical", "and", "chemical", "signals"],
    ["the", "immune", "system", "protects", "the", "body", "from", "disease"],
    ["ecosystems", "contain", "producers", "consumers", "and", "decomposers"],
    ["photosynthesis", "converts", "light", "into", "chemical", "energy"],
    ["respiration", "releases", "energy", "from", "glucose"],
    ["mitosis", "produces", "identical", "daughter", "cells"],
    ["meiosis", "produces", "genetically", "diverse", "sex", "cells"],
    ["mutations", "change", "the", "DNA", "sequence"],
    ["natural", "selection", "favours", "traits", "that", "improve", "survival"],
    ["adaptation", "allows", "organisms", "to", "fit", "their", "environment"],
    ["biodiversity", "is", "the", "variety", "of", "life", "on", "earth"],
    # ── Neuroscience / cognition ───────────────────────────────────────────
    ["memory", "is", "stored", "in", "patterns", "of", "neural", "activity"],
    ["attention", "selects", "relevant", "information", "for", "processing"],
    ["learning", "changes", "the", "strength", "of", "synaptic", "connections"],
    ["the", "prefrontal", "cortex", "is", "involved", "in", "planning"],
    ["emotion", "influences", "decision", "making", "and", "memory"],
    ["sleep", "consolidates", "memories", "and", "clears", "waste", "products"],
    ["language", "is", "processed", "in", "specialised", "brain", "regions"],
    ["perception", "involves", "both", "bottom-up", "and", "top-down", "processing"],
    ["consciousness", "is", "associated", "with", "global", "workspace", "activity"],
    ["the", "hippocampus", "is", "essential", "for", "forming", "new", "memories"],
    ["dopamine", "is", "a", "neurotransmitter", "involved", "in", "reward"],
    ["stress", "hormones", "affect", "learning", "and", "memory"],
    # ── Psychology / philosophy ────────────────────────────────────────────
    ["beliefs", "guide", "behaviour", "and", "interpretation"],
    ["emotions", "provide", "rapid", "evaluations", "of", "situations"],
    ["reasoning", "can", "be", "fast", "intuitive", "or", "slow", "deliberate"],
    ["cognitive", "biases", "distort", "judgment", "and", "perception"],
    ["motivation", "drives", "goal-directed", "behaviour"],
    ["social", "norms", "regulate", "group", "behaviour"],
    ["language", "shapes", "how", "we", "think", "about", "the", "world"],
    ["meaning", "arises", "from", "the", "use", "of", "symbols", "in", "context"],
    ["perception", "is", "an", "active", "constructive", "process"],
    ["memory", "is", "reconstructive", "not", "reproductive"],
    ["thinking", "involves", "mental", "representations", "and", "operations"],
    ["intelligence", "involves", "learning", "reasoning", "and", "adaptation"],
    # ── Mathematics ───────────────────────────────────────────────────────
    ["numbers", "represent", "quantities", "and", "measurements"],
    ["addition", "and", "subtraction", "are", "inverse", "operations"],
    ["multiplication", "is", "repeated", "addition"],
    ["probability", "measures", "the", "likelihood", "of", "events"],
    ["statistics", "summarises", "and", "interprets", "data"],
    ["functions", "map", "inputs", "to", "outputs"],
    ["graphs", "represent", "relationships", "between", "variables"],
    ["logic", "studies", "valid", "patterns", "of", "reasoning"],
    ["sets", "are", "collections", "of", "distinct", "objects"],
    ["algorithms", "are", "step-by-step", "procedures", "for", "solving", "problems"],
    ["recursion", "is", "a", "process", "that", "calls", "itself"],
    ["proof", "establishes", "the", "truth", "of", "mathematical", "statements"],
    # ── Computer science / AI ─────────────────────────────────────────────
    ["programs", "are", "instructions", "that", "computers", "execute"],
    ["data", "structures", "organise", "information", "for", "efficient", "access"],
    ["sorting", "algorithms", "arrange", "elements", "in", "order"],
    ["machine", "learning", "finds", "patterns", "in", "data"],
    ["neural", "networks", "are", "inspired", "by", "the", "brain"],
    ["natural", "language", "processing", "helps", "computers", "understand", "text"],
    ["search", "algorithms", "find", "solutions", "in", "large", "spaces"],
    ["databases", "store", "and", "retrieve", "structured", "information"],
    ["networks", "connect", "computers", "and", "allow", "communication"],
    ["cryptography", "protects", "data", "from", "unauthorised", "access"],
    ["compilers", "translate", "source", "code", "into", "machine", "code"],
    ["abstraction", "hides", "complexity", "behind", "simple", "interfaces"],
    # ── Synonyms / near-synonyms (build similarity) ────────────────────────
    ["big", "large", "huge", "enormous", "vast", "immense", "giant"],
    ["small", "little", "tiny", "minute", "microscopic", "miniature"],
    ["fast", "quick", "rapid", "swift", "speedy", "prompt"],
    ["slow", "sluggish", "gradual", "unhurried", "leisurely"],
    ["happy", "joyful", "glad", "pleased", "content", "cheerful"],
    ["sad", "unhappy", "sorrowful", "melancholy", "depressed"],
    ["hot", "warm", "heated", "burning", "scorching"],
    ["cold", "cool", "chilly", "freezing", "icy"],
    ["start", "begin", "commence", "initiate", "launch"],
    ["end", "finish", "conclude", "terminate", "stop", "complete"],
    ["make", "create", "build", "construct", "produce", "generate", "form"],
    ["show", "display", "exhibit", "demonstrate", "reveal", "present"],
    ["say", "tell", "speak", "state", "declare", "announce", "mention"],
    ["think", "believe", "consider", "suppose", "reckon", "assume"],
    ["know", "understand", "realise", "recognise", "comprehend"],
    ["want", "need", "desire", "wish", "require", "seek"],
    ["help", "assist", "support", "aid", "facilitate"],
    ["use", "employ", "utilise", "apply", "leverage"],
    ["find", "discover", "locate", "detect", "identify"],
    ["move", "travel", "go", "proceed", "advance", "transfer"],
    # ── Antonyms (build contrast) ──────────────────────────────────────────
    ["hot", "and", "cold", "are", "opposite", "temperatures"],
    ["big", "and", "small", "are", "opposite", "sizes"],
    ["fast", "and", "slow", "are", "opposite", "speeds"],
    ["light", "and", "dark", "are", "opposite", "illumination", "levels"],
    ["hard", "and", "soft", "are", "opposite", "textures"],
    ["strong", "and", "weak", "are", "opposite", "strengths"],
    ["young", "and", "old", "describe", "opposite", "ages"],
    ["open", "and", "closed", "are", "opposite", "states"],
    ["true", "and", "false", "are", "opposite", "truth", "values"],
    ["cause", "and", "effect", "are", "related", "events"],
    # ── Causal chains ─────────────────────────────────────────────────────
    ["rain", "causes", "flooding", "which", "damages", "infrastructure"],
    ["exercise", "improves", "health", "and", "reduces", "disease", "risk"],
    ["pollution", "causes", "climate", "change", "and", "health", "problems"],
    ["education", "increases", "income", "and", "reduces", "poverty"],
    ["stress", "causes", "illness", "by", "suppressing", "the", "immune", "system"],
    ["fire", "needs", "fuel", "oxygen", "and", "heat", "to", "burn"],
    ["deforestation", "causes", "habitat", "loss", "and", "climate", "change"],
    ["vaccination", "prevents", "disease", "by", "training", "the", "immune", "system"],
    # ── Hierarchies (is-a relationships) ──────────────────────────────────
    ["a", "dog", "is", "a", "mammal", "which", "is", "an", "animal"],
    ["a", "rose", "is", "a", "flower", "which", "is", "a", "plant"],
    ["a", "car", "is", "a", "vehicle", "which", "is", "a", "machine"],
    ["a", "hammer", "is", "a", "tool", "used", "for", "building"],
    ["gold", "is", "a", "metal", "which", "is", "an", "element"],
    ["a", "river", "is", "a", "body", "of", "water"],
    ["a", "novel", "is", "a", "long", "work", "of", "fiction"],
    ["a", "violin", "is", "a", "stringed", "musical", "instrument"],
    # ── Properties / attributes ────────────────────────────────────────────
    ["water", "is", "liquid", "at", "room", "temperature"],
    ["ice", "is", "the", "solid", "form", "of", "water"],
    ["steam", "is", "the", "gaseous", "form", "of", "water"],
    ["gold", "is", "a", "shiny", "dense", "valuable", "metal"],
    ["glass", "is", "transparent", "and", "fragile"],
    ["rubber", "is", "elastic", "and", "waterproof"],
    ["wood", "is", "a", "strong", "light", "natural", "material"],
    ["steel", "is", "a", "strong", "durable", "metal", "alloy"],
    # ── Social / human activity ────────────────────────────────────────────
    ["governments", "make", "laws", "and", "manage", "public", "services"],
    ["economies", "involve", "the", "production", "distribution", "and", "consumption", "of", "goods"],
    ["trade", "allows", "specialisation", "and", "increases", "prosperity"],
    ["communication", "allows", "people", "to", "share", "ideas", "and", "coordinate"],
    ["culture", "is", "shared", "beliefs", "values", "and", "practices"],
    ["religion", "provides", "meaning", "community", "and", "moral", "guidance"],
    ["science", "advances", "through", "hypothesis", "testing", "and", "peer", "review"],
    ["medicine", "treats", "disease", "and", "promotes", "health"],
    ["agriculture", "produces", "food", "for", "human", "populations"],
    ["transportation", "moves", "people", "and", "goods", "between", "places"],
]


class DistributionalCodebook:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._codebook: Dict[str, HyperVector] = {}

    def build_from_corpus(self, sentences: List[List[str]]) -> None:
        word_contexts: Dict[str, List[HyperVector]] = {}
        for sentence in sentences:
            for i, word in enumerate(sentence):
                w = word.lower()
                if w not in word_contexts:
                    word_contexts[w] = []
                for offset in range(-self.window_size, self.window_size + 1):
                    if offset == 0:
                        continue
                    j = i + offset
                    if 0 <= j < len(sentence):
                        context_word = sentence[j].lower()
                        ctx_hv = HyperVector(hash(context_word) % (2**32))
                        # Permute by offset to encode position (silently skip if unsupported)
                        try:
                            ctx_hv = ctx_hv.permute(offset)
                        except (AttributeError, TypeError) as e:
                            logger.debug("permute unavailable, using unpermuted HV: %s", e)
                        word_contexts[w].append(ctx_hv)
        for word, context_hvs in word_contexts.items():
            if not context_hvs:
                self._codebook[word] = HyperVector(hash(word) % (2**32))
                continue
            accumulated = context_hvs[0]
            for hv in context_hvs[1:]:
                accumulated = accumulated.bundle(hv)
            self._codebook[word] = accumulated

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"window_size": self.window_size, "codebook": self._codebook}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.window_size = data["window_size"]
        self._codebook = data["codebook"]

    def get_hv(self, word: str) -> Optional[HyperVector]:
        return self._codebook.get(word.lower())

    def similarity(self, word1: str, word2: str) -> float:
        hv1 = self.get_hv(word1)
        hv2 = self.get_hv(word2)
        if hv1 is None or hv2 is None:
            return 0.0
        try:
            return float(hv1.similarity(hv2))
        except Exception:
            return 0.0

    def online_observe(self, sentence: List[str]) -> None:
        """
        Update the codebook *incrementally* from a single new sentence.

        Instead of re-building from scratch, this bundles new context HVs
        into the existing codebook entries using majority-vote bundling —
        equivalent to a single pass of :meth:`build_from_corpus` for that
        sentence.

        This is O(|sentence| * window_size) per call, making it suitable for
        real-time learning from a stream of text.  Conceptually grounded in
        *online* / *streaming* learning (Bottou, 1998) and the Hebbian
        principle: "what fires together wires together".

        Args:
            sentence:  List of tokens (words) to learn from.
        """
        for i, word in enumerate(sentence):
            w = word.lower()
            for offset in range(-self.window_size, self.window_size + 1):
                if offset == 0:
                    continue
                j = i + offset
                if 0 <= j < len(sentence):
                    context_word = sentence[j].lower()
                    ctx_hv = HyperVector(hash(context_word) % (2**32))
                    try:
                        ctx_hv = ctx_hv.permute(offset)
                    except (AttributeError, TypeError):
                        pass
                    if w in self._codebook:
                        self._codebook[w] = self._codebook[w].bundle(ctx_hv)
                    else:
                        self._codebook[w] = ctx_hv

    @classmethod
    def build_default(cls, window_size: int = 3) -> "DistributionalCodebook":
        cb = cls(window_size=window_size)
        cb.build_from_corpus(BUILTIN_CORPUS)
        return cb
