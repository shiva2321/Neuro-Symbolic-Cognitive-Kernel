"""Distributional Semantics via VSA context windows for NSCK V3."""
from __future__ import annotations
import logging
import pickle
from typing import List, Dict, Optional
import python.core.vsa.hypervec_shim as hypervec_rs

logger = logging.getLogger(__name__)

HyperVector = hypervec_rs.HyperVector

BUILTIN_CORPUS: List[List[str]] = [
    # ── Natural world ────────────────────────────────────────────────────────
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "in", "the", "park"],
    ["cats", "and", "dogs", "are", "common", "pets"],
    ["the", "sun", "shines", "in", "the", "sky"],
    ["water", "flows", "in", "rivers", "and", "oceans"],
    ["plants", "need", "water", "and", "sunlight", "to", "grow"],
    ["animals", "eat", "food", "to", "survive"],
    ["humans", "are", "social", "animals"],
    ["the", "earth", "orbits", "the", "sun"],
    ["the", "moon", "orbits", "the", "earth"],
    ["gravity", "pulls", "objects", "toward", "the", "earth"],
    ["light", "travels", "faster", "than", "sound"],
    ["energy", "cannot", "be", "created", "or", "destroyed"],
    ["atoms", "make", "up", "all", "matter"],
    ["cells", "are", "the", "basic", "unit", "of", "life"],
    ["genes", "carry", "genetic", "information"],
    ["evolution", "explains", "the", "diversity", "of", "life"],
    ["forests", "provide", "oxygen", "and", "shelter"],
    ["the", "ocean", "covers", "most", "of", "the", "earth"],
    ["rivers", "flow", "from", "mountains", "to", "the", "sea"],
    ["clouds", "form", "from", "water", "vapor", "in", "the", "atmosphere"],
    ["rain", "falls", "from", "clouds", "to", "the", "ground"],
    ["birds", "can", "fly", "because", "they", "have", "wings"],
    ["fish", "live", "and", "breathe", "in", "water"],
    ["trees", "convert", "carbon", "dioxide", "into", "oxygen"],
    ["the", "climate", "affects", "living", "conditions"],
    ["food", "provides", "energy", "for", "living", "organisms"],
    ["water", "is", "essential", "for", "life"],
    ["oxygen", "is", "needed", "for", "breathing"],
    ["the", "heart", "pumps", "blood", "through", "the", "body"],
    ["the", "lungs", "exchange", "oxygen", "and", "carbon", "dioxide"],
    ["the", "brain", "controls", "the", "nervous", "system"],
    ["the", "sun", "provides", "energy", "for", "life", "on", "earth"],
    ["fire", "requires", "oxygen", "fuel", "and", "heat"],
    ["ice", "melts", "when", "temperature", "rises", "above", "zero"],
    ["snow", "falls", "when", "temperature", "drops", "below", "freezing"],
    ["earthquakes", "occur", "when", "tectonic", "plates", "shift"],
    ["volcanoes", "erupt", "and", "release", "lava", "and", "gas"],
    ["soil", "contains", "minerals", "and", "organic", "matter"],
    ["seeds", "germinate", "when", "conditions", "are", "suitable"],
    # ── Science + technology ─────────────────────────────────────────────────
    ["science", "studies", "the", "natural", "world"],
    ["physics", "and", "chemistry", "are", "natural", "sciences"],
    ["biology", "studies", "living", "organisms"],
    ["mathematics", "is", "the", "language", "of", "science"],
    ["computers", "process", "data", "and", "information"],
    ["the", "internet", "connects", "people", "worldwide"],
    ["technology", "changes", "human", "society"],
    ["algorithms", "are", "step", "by", "step", "instructions"],
    ["software", "controls", "hardware", "through", "programs"],
    ["data", "is", "stored", "in", "memory", "and", "databases"],
    ["networks", "enable", "communication", "between", "devices"],
    ["sensors", "detect", "physical", "signals", "from", "the", "environment"],
    ["machine", "learning", "identifies", "patterns", "in", "data"],
    ["artificial", "intelligence", "simulates", "human", "reasoning"],
    ["neural", "networks", "are", "inspired", "by", "the", "brain"],
    ["electricity", "flows", "through", "conductors"],
    ["magnets", "attract", "iron", "and", "steel"],
    ["light", "is", "both", "a", "wave", "and", "a", "particle"],
    ["heat", "flows", "from", "hot", "to", "cold", "objects"],
    ["chemical", "reactions", "produce", "new", "substances"],
    # ── Society + humans ─────────────────────────────────────────────────────
    ["language", "enables", "communication", "between", "humans"],
    ["knowledge", "is", "power", "and", "freedom"],
    ["cities", "are", "large", "human", "settlements"],
    ["history", "records", "past", "human", "events"],
    ["art", "expresses", "human", "emotion", "and", "culture"],
    ["music", "and", "art", "are", "creative", "activities"],
    ["the", "cat", "chased", "the", "mouse", "across", "the", "room"],
    ["students", "learn", "new", "skills", "at", "school"],
    ["teachers", "help", "students", "understand", "concepts"],
    ["doctors", "treat", "patients", "with", "medicine"],
    ["engineers", "design", "and", "build", "structures"],
    ["laws", "govern", "human", "behavior", "in", "society"],
    ["money", "is", "used", "to", "exchange", "goods", "and", "services"],
    ["families", "provide", "support", "and", "care", "for", "members"],
    ["governments", "manage", "resources", "and", "provide", "services"],
    ["trade", "enables", "exchange", "of", "goods", "between", "regions"],
    ["education", "develops", "knowledge", "and", "skills"],
    ["medicine", "prevents", "and", "treats", "diseases"],
    ["sports", "promote", "health", "and", "competition"],
    ["writing", "preserves", "knowledge", "across", "generations"],
    # ── Cognition + mind ─────────────────────────────────────────────────────
    ["memory", "stores", "past", "experiences", "and", "knowledge"],
    ["attention", "focuses", "cognitive", "resources", "on", "a", "task"],
    ["perception", "interprets", "signals", "from", "the", "senses"],
    ["reasoning", "draws", "conclusions", "from", "evidence"],
    ["learning", "changes", "behavior", "through", "experience"],
    ["creativity", "combines", "ideas", "in", "new", "ways"],
    ["emotion", "influences", "decision", "making", "and", "behavior"],
    ["consciousness", "is", "awareness", "of", "self", "and", "environment"],
    ["language", "shapes", "thought", "and", "understanding"],
    ["problem", "solving", "requires", "goal", "definition", "and", "planning"],
    ["beliefs", "guide", "actions", "and", "interpretations"],
    ["fear", "triggers", "a", "fight", "or", "flight", "response"],
    ["sleep", "consolidates", "memory", "and", "restores", "the", "brain"],
    ["stress", "can", "impair", "cognitive", "function"],
    ["practice", "improves", "skill", "through", "repetition"],
    # ── Cause and effect ─────────────────────────────────────────────────────
    ["rain", "causes", "rivers", "to", "rise"],
    ["heat", "causes", "water", "to", "evaporate"],
    ["exercise", "strengthens", "muscles", "and", "improves", "health"],
    ["pollution", "damages", "ecosystems", "and", "health"],
    ["deforestation", "reduces", "biodiversity", "and", "increases", "erosion"],
    ["smoking", "causes", "lung", "disease", "and", "cancer"],
    ["overeating", "leads", "to", "weight", "gain"],
    ["lack", "of", "sleep", "impairs", "concentration"],
    ["sunlight", "enables", "photosynthesis", "in", "plants"],
    ["cold", "temperatures", "slow", "chemical", "reactions"],
    ["high", "pressure", "increases", "the", "boiling", "point", "of", "water"],
    ["gravity", "causes", "objects", "to", "fall"],
    ["friction", "slows", "moving", "objects"],
    ["wind", "erodes", "rock", "over", "time"],
    ["infection", "triggers", "an", "immune", "response"],
    # ── Properties + descriptions ─────────────────────────────────────────────
    ["water", "is", "transparent", "and", "odorless"],
    ["gold", "is", "a", "dense", "and", "malleable", "metal"],
    ["glass", "is", "brittle", "and", "transparent"],
    ["rubber", "is", "elastic", "and", "waterproof"],
    ["iron", "is", "magnetic", "and", "conducts", "electricity"],
    ["salt", "dissolves", "in", "water"],
    ["oil", "does", "not", "mix", "with", "water"],
    ["wood", "is", "a", "renewable", "and", "biodegradable", "material"],
    ["plastic", "is", "durable", "but", "not", "biodegradable"],
    ["copper", "is", "an", "excellent", "conductor", "of", "electricity"],
    ["air", "is", "a", "mixture", "of", "gases"],
    ["sound", "travels", "through", "vibrations", "in", "a", "medium"],
    ["pressure", "is", "force", "per", "unit", "area"],
    ["velocity", "is", "speed", "in", "a", "specific", "direction"],
    ["density", "is", "mass", "per", "unit", "volume"],
    # ── Time + sequence ──────────────────────────────────────────────────────
    ["spring", "comes", "before", "summer"],
    ["autumn", "follows", "summer"],
    ["morning", "comes", "before", "afternoon"],
    ["childhood", "precedes", "adulthood"],
    ["cause", "precedes", "effect"],
    ["birth", "happens", "before", "death"],
    ["planning", "comes", "before", "execution"],
    ["design", "precedes", "construction"],
    ["diagnosis", "precedes", "treatment"],
    ["research", "comes", "before", "publication"],
    ["input", "is", "processed", "before", "output", "is", "produced"],
    ["a", "seed", "grows", "into", "a", "plant", "over", "time"],
    ["stars", "form", "from", "clouds", "of", "gas", "and", "dust"],
    ["mountains", "form", "over", "millions", "of", "years"],
    ["languages", "change", "gradually", "over", "time"],
    # ── Similarity + difference ──────────────────────────────────────────────
    ["cats", "and", "dogs", "are", "both", "mammals"],
    ["birds", "and", "bats", "both", "fly", "but", "are", "not", "related"],
    ["plants", "and", "animals", "are", "both", "living", "organisms"],
    ["heat", "and", "temperature", "are", "related", "but", "different"],
    ["speed", "and", "velocity", "differ", "in", "that", "velocity", "has", "direction"],
    ["sugar", "and", "salt", "are", "both", "white", "crystals"],
    ["the", "moon", "and", "the", "sun", "both", "appear", "in", "the", "sky"],
    ["wolves", "and", "dogs", "share", "a", "common", "ancestor"],
    ["cold", "and", "hot", "are", "opposite", "temperature", "extremes"],
    ["night", "and", "day", "are", "caused", "by", "earth", "rotation"],
    # ── Negation + exceptions ────────────────────────────────────────────────
    ["not", "all", "birds", "can", "fly"],
    ["whales", "are", "mammals", "not", "fish"],
    ["bats", "are", "mammals", "not", "birds"],
    ["pluto", "is", "not", "classified", "as", "a", "planet"],
    ["viruses", "are", "not", "living", "organisms"],
    ["glass", "is", "not", "a", "crystal"],
    ["tomatoes", "are", "fruits", "not", "vegetables"],
    ["humans", "cannot", "photosynthesize"],
    ["sound", "cannot", "travel", "through", "a", "vacuum"],
    ["water", "does", "not", "burn"],
    # ── Spatial + structural ─────────────────────────────────────────────────
    ["the", "nucleus", "is", "at", "the", "center", "of", "the", "atom"],
    ["the", "heart", "is", "located", "in", "the", "chest"],
    ["the", "cerebellum", "is", "at", "the", "back", "of", "the", "brain"],
    ["roots", "grow", "below", "the", "soil"],
    ["leaves", "grow", "above", "the", "stem"],
    ["the", "sun", "is", "above", "the", "horizon", "during", "the", "day"],
    ["the", "moon", "is", "closer", "to", "earth", "than", "the", "sun"],
    ["mountains", "are", "taller", "than", "hills"],
    ["oceans", "are", "deeper", "than", "lakes"],
    ["the", "core", "of", "the", "earth", "is", "hotter", "than", "the", "surface"],
    # ── Analogy + abstract ───────────────────────────────────────────────────
    ["the", "brain", "is", "like", "a", "computer"],
    ["the", "heart", "functions", "like", "a", "pump"],
    ["the", "eye", "works", "like", "a", "camera"],
    ["a", "cell", "is", "like", "a", "miniature", "city"],
    ["memory", "is", "like", "a", "library"],
    ["learning", "is", "like", "building", "a", "mental", "map"],
    ["a", "concept", "is", "like", "a", "node", "in", "a", "network"],
    ["reasoning", "is", "like", "navigation", "through", "knowledge"],
    ["language", "is", "a", "tool", "for", "sharing", "thoughts"],
    ["a", "hypothesis", "is", "a", "testable", "prediction"],
]


class DistributionalCodebook:
    def __init__(self, window_size: int = 5, pretrain: bool = True):
        self.window_size = window_size
        self._codebook: Dict[str, HyperVector] = {}
        # Pre-train on the built-in corpus so semantic similarity is meaningful
        # from the very first sentence processed (rather than random-hash fallback).
        if pretrain:
            self.build_from_corpus(BUILTIN_CORPUS)

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

    @classmethod
    def build_default(cls, window_size: int = 3, enable_hf_corpus: bool = False) -> "DistributionalCodebook":
        """Build a codebook pre-trained on the built-in corpus (and optionally HuggingFace data).

        Parameters
        ----------
        window_size     : context window radius for co-occurrence bundling.
        enable_hf_corpus: when True, attempt to download a slice of
                          ``fka/awesome-chatgpt-prompts`` to augment the
                          built-in sentences.  Falls back silently.
        """
        cb = cls(window_size=window_size, pretrain=False)
        if enable_hf_corpus:
            try:
                from python.core.language.hf_corpus_loader import load_hf_corpus
                hf_sents = load_hf_corpus("auto", max_sentences=2000)
                if hf_sents:
                    corpus = list(BUILTIN_CORPUS) + hf_sents
                    logger.info("[DistribCodebook] Extended corpus with %d HF sentences "
                                "(total %d)", len(hf_sents), len(corpus))
                else:
                    corpus = BUILTIN_CORPUS
            except Exception as exc:
                logger.debug("[DistribCodebook] HF corpus load failed: %s", exc)
                corpus = BUILTIN_CORPUS
        else:
            corpus = BUILTIN_CORPUS
        cb.build_from_corpus(corpus)
        return cb
