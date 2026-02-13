"""
NSCK Phase 8: Universal Input Layer
====================================
Maps arbitrary sensor data (scalars, categories, dicts, lists) into the
10,240-bit binary VSA hypervector space so that every modality shares
the same algebraic structure (XOR=bind, bundle=superposition, permute=sequence).

Design decisions
----------------
* **Scalars** use Thermometer Encoding (quantise into N bins → deterministic
  HVs → weighted bundle by proximity).  Preserves the *Scalar Similarity
  Test*: nearby values produce high cosine overlap.
* **Categoricals** are codebook entries with deterministic seeding per
  ``domain:label`` pair.  An LRU cache (max_size) prevents codebook explosion.
* **Dicts** use recursive role-filler binding: Role_HV ⊗ Value_HV for each
  key, then XOR-bundle all pairs.
* **Lists** use permutation-based sequence encoding: A ⊕ ρ(B) ⊕ ρ²(C).
* **Text** uses a 4-component architecture:
    1. Keyword HV (topical similarity via shared content words)
    2. Character n-gram HV (sub-word / morphological overlap)
    3. Word-order HV (positional permutation encoding)
    4. **Phrase-structure HV** (compositional parse via VSA binding)
  Components are combined by segment concatenation (50/15/15/20 split).
"""

from __future__ import annotations

import hashlib
import collections
from typing import Any, Dict, Optional, Tuple

import numpy as np
import hypervec_shim as hv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIMENSION = 10240
DEFAULT_THERMOMETER_BINS = 100
MAX_CODEBOOK_SIZE = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stable_seed(label: str) -> int:
    """Deterministic 64-bit seed from an arbitrary string label."""
    h = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") & 0xFFFF_FFFF  # u32 to match HV ctor


def _deterministic_bundle(hvs: list) -> "hv.HyperVector":
    """Majority-vote bundle with deterministic tiebreaking.

    Unlike `HyperVector.bundle()` which uses random tiebreak (adding noise),
    this uses a fixed seed derived from the input HVs themselves.  For an
    odd number of inputs, there are no ties.  For even numbers, the first
    input's bits are used as the tiebreaker.

    This ensures that the *same set of keywords* always produces the
    *exact same* HV, making similarity measurements reliable.
    """
    if len(hvs) == 0:
        return hv.HyperVector(0)
    if len(hvs) == 1:
        return hvs[0]

    # Stack all bit arrays
    bits = np.stack([h.bits for h in hvs], axis=0)  # shape: (N, D)
    vote_sum = bits.sum(axis=0)                       # shape: (D,)
    majority = len(hvs) / 2.0

    result = np.zeros(bits.shape[1], dtype=np.int8)
    result[vote_sum > majority] = 1
    # Tiebreak: use first HV's bits (deterministic)
    ties = vote_sum == majority
    result[ties] = hvs[0].bits[ties]

    return hv.HyperVector.from_bits(result)


# ---------------------------------------------------------------------------
# Phrase-Structure Parser (symbolic, no neural nets)
# ---------------------------------------------------------------------------
# Lightweight regex-based chunker that identifies NP, VP, PP phrases,
# then composes them using VSA role-filler binding to produce a
# *compositional* HV encoding the parse tree.
#
# Grammar (simplified):
#   S    → NP VP (PP)*
#   NP   → (Det)? (Adj)* N
#   VP   → V (NP)?
#   PP   → Prep NP
#
# Encoding strategy (pure VSA algebra):
#   NP_hv   = bind(ROLE_det, det_hv) ⊕ bind(ROLE_adj, adj_hv) ⊕ bind(ROLE_head, noun_hv)
#   VP_hv   = bind(ROLE_verb, verb_hv) ⊕ bind(ROLE_obj, NP_obj_hv)
#   S_hv    = bind(ROLE_subj, NP_subj_hv) ⊕ bind(ROLE_pred, VP_hv) ⊕ bind(ROLE_pp, PP_hv)
# ---------------------------------------------------------------------------

# POS tag sets (closed-class words for chunking)
_DETERMINERS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "my", "your",
    "his", "her", "its", "our", "their", "some", "any", "no", "every",
    "each", "all", "both", "few", "many", "several", "much",
})

_PREPOSITIONS = frozenset({
    "in", "on", "at", "by", "for", "with", "from", "to", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "under", "over", "about", "against", "among", "around", "behind",
    "beyond", "near", "of", "off", "out", "up", "upon", "using",
})

_ADJECTIVE_SUFFIXES = ("ful", "less", "ous", "ive", "ble", "ial", "al",
                       "ent", "ant", "ing", "ed", "ic", "ary", "ory")

_BE_VERBS = frozenset({"is", "are", "was", "were", "be", "been", "being"})

_AUX_VERBS = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "shall", "can", "may", "might", "must",
})

# ---------------------------------------------------------------------------
# Semantic Role System
# ---------------------------------------------------------------------------
# Semantic roles capture *functional* relationships independent of syntax:
#   Agent:      volitional actor ("John broke the vase")
#   Patient:    affected entity ("the vase broke")
#   Experiencer: psychological participant ("Mary heard music")
#   Theme:      moved/located entity ("Bob put the book on the shelf")
#   Instrument: means of action ("She cut with a knife")
#   Location:   spatial setting ("The meeting is in the office")
#   Source:     origin ("He came from Boston")
#   Goal:       destination ("They went to Paris")
# ---------------------------------------------------------------------------

# Syntactic role HVs (deterministic, shared globally)
_ROLE_CACHE: dict = {}


def _role_hv(role_name: str):
    """Get a deterministic role HV for a syntactic or semantic function.
    
    Supports both syntactic roles (subj, obj, pred) and semantic roles
    (agent, patient, instrument, theme, experiencer, location, source, goal).
    """
    if role_name not in _ROLE_CACHE:
        _ROLE_CACHE[role_name] = hv.HyperVector(
            _stable_seed(f"__role__{role_name}")
        )
    return _ROLE_CACHE[role_name]


# Verb classification for semantic role assignment
_AGENTIVE_VERBS = {
    "break", "build", "create", "destroy", "make", "write", "paint",
    "cut", "hit", "push", "pull", "throw", "kick", "open", "close",
    "eat", "drink", "cook", "clean", "wash", "fix", "repair",
    # Irregular past tense forms
    "broke", "built", "made", "wrote", "cut", "hit"
}

_EXPERIENCER_VERBS = {
    "hear", "see", "feel", "smell", "taste", "know", "believe",
    "think", "understand", "remember", "forget", "like", "love",
    "hate", "want", "need", "prefer", "enjoy", "fear", "hope",
    # Irregular past tense forms
    "heard", "saw", "felt", "knew", "thought", "understood",
    "remembered", "forgot", "liked", "loved", "hated", "wanted"
}

_MOTION_VERBS = {
    "go", "come", "move", "run", "walk", "fly", "swim", "travel",
    "arrive", "depart", "enter", "exit", "leave", "return", "roll",
    # Irregular past tense forms
    "went", "came", "moved", "ran", "walked", "flew", "swam", "traveled",
    "arrived", "departed", "entered", "exited", "left", "returned", "rolled"
}

_TRANSFER_VERBS = {
    "give", "send", "bring", "take", "carry", "deliver", "pass",
    "hand", "offer", "show", "teach", "tell",
    # Irregular past tense forms
    "gave", "sent", "brought", "took", "carried", "delivered",
    "passed", "handed", "offered", "showed", "taught", "told"
}

_POSITIONAL_VERBS = {
    "put", "place", "set", "lay", "stand", "sit", "hang", "install",
    # Past tense forms
    "placed", "laid", "stood", "sat", "hung", "installed"
}


def _classify_verb_semantics(verb: str) -> str:
    """Classify verb into semantic categories for role assignment.
    
    Returns one of: agentive, experiencer, motion, transfer, positional, stative.
    """
    v = verb.lower()
    
    # Try exact match first
    if v in _AGENTIVE_VERBS:
        return "agentive"
    if v in _EXPERIENCER_VERBS:
        return "experiencer"
    if v in _MOTION_VERBS:
        return "motion"
    if v in _TRANSFER_VERBS:
        return "transfer"
    if v in _POSITIONAL_VERBS:
        return "positional"
    
    # Try stemming: remove common inflections
    # Past tense: -ed
    if v.endswith("ed") and len(v) > 3:
        stem = v[:-2]
        if stem in _AGENTIVE_VERBS or stem in _EXPERIENCER_VERBS:
            if stem in _AGENTIVE_VERBS:
                return "agentive"
            if stem in _EXPERIENCER_VERBS:
                return "experiencer"
    
    # Present 3rd person: -s/-es
    if v.endswith("es") and len(v) > 3:
        stem = v[:-2]
        if stem in _AGENTIVE_VERBS or stem in _EXPERIENCER_VERBS or stem in _MOTION_VERBS:
            return _classify_verb_semantics(stem)
    elif v.endswith("s") and len(v) > 2 and not v.endswith("ss"):
        stem = v[:-1]
        if stem in _AGENTIVE_VERBS or stem in _EXPERIENCER_VERBS or stem in _MOTION_VERBS:
            return _classify_verb_semantics(stem)
    
    # Present participle: -ing
    if v.endswith("ing") and len(v) > 4:
        stem = v[:-3]
        if stem in _AGENTIVE_VERBS or stem in _EXPERIENCER_VERBS or stem in _MOTION_VERBS:
            return _classify_verb_semantics(stem)
        # Try doubling consonant removal (running -> run)
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem2 = stem[:-1]
            if stem2 in _AGENTIVE_VERBS or stem2 in _MOTION_VERBS:
                return _classify_verb_semantics(stem2)
    
    return "stative"  # Default for be/have/become and unknown verbs


def _assign_semantic_roles(parsed: dict) -> dict:
    """Map syntactic structure to semantic roles based on verb semantics.
    
    Takes output from _chunk_phrases() and returns a dict mapping
    semantic roles (agent, patient, theme, etc.) to their fillers.
    
    Example:
        Input:  {subject_np: {head: "John"}, verb: "broke", object_np: {head: "vase"}}
        Output: {"agent": {head: "John"}, "patient": {head: "vase"}, "action": "broke"}
    """
    roles = {}
    
    if not parsed.get("verb"):
        return roles
    
    verb = parsed["verb"]
    verb_class = _classify_verb_semantics(verb)
    roles["action"] = verb
    roles["verb_class"] = verb_class
    
    subj = parsed.get("subject_np")
    obj = parsed.get("object_np")
    
    # Semantic role assignment based on verb class
    if verb_class == "agentive":
        # Subject is volitional agent, object is affected patient
        if subj:
            roles["agent"] = subj
        if obj:
            roles["patient"] = obj
    
    elif verb_class == "experiencer":
        # Subject is experiencer (psychological participant), object is stimulus
        if subj:
            roles["experiencer"] = subj
        if obj:
            roles["theme"] = obj  # What is perceived/thought about
    
    elif verb_class == "motion":
        # Subject is theme (thing moving)
        if subj:
            roles["theme"] = subj
    
    elif verb_class == "transfer":
        # Subject is agent, object is theme (thing transferred)
        if subj:
            roles["agent"] = subj
        if obj:
            roles["theme"] = obj
    
    elif verb_class == "positional":
        # Subject is agent, object is theme (thing positioned)
        if subj:
            roles["agent"] = subj
        if obj:
            roles["theme"] = obj
    
    else:  # stative
        # Subject is just subject (no agency)
        if subj:
            roles["subject"] = subj
        if obj:
            roles["object"] = obj
    
    # Process prepositional phrases for location/instrument/source/goal
    for pp in parsed.get("prep_phrases", []):
        prep = pp["prep"].lower()
        np = pp["np"]
        
        if prep in ("with", "using", "by"):
            roles["instrument"] = np
        elif prep in ("in", "at", "on", "near", "above", "below", "beside"):
            roles["location"] = np
        elif prep in ("from", "out"):
            roles["source"] = np
        elif prep in ("to", "into", "toward"):
            roles["goal"] = np
        else:
            # Generic prepositional modifier
            if "modifier" not in roles:
                roles["modifier"] = []
            roles["modifier"].append({"prep": prep, "np": np})
    
    return roles


def _pos_tag_simple(word: str) -> str:
    """Heuristic POS tagger (no model, no data files).

    Returns one of: DET, PREP, ADJ, VERB, NOUN, AUX, CONJ, ADV, PRON, UNK.
    """
    w = word.lower()
    if w in _DETERMINERS:
        return "DET"
    if w in _PREPOSITIONS:
        return "PREP"
    if w in _BE_VERBS:
        return "AUX"
    if w in _AUX_VERBS:
        return "AUX"
    if w in ("and", "or", "but", "nor", "yet", "so"):
        return "CONJ"
    if w in ("i", "me", "you", "he", "she", "it", "we", "they",
             "him", "her", "us", "them", "who", "what", "which"):
        return "PRON"
    if w in ("very", "quite", "rather", "really", "always", "never",
             "often", "usually", "sometimes", "here", "there", "now",
             "then", "also", "already", "still", "just", "only", "not"):
        return "ADV"
    
    # Check verb lexicon (includes irregular forms)
    if (w in _AGENTIVE_VERBS or w in _EXPERIENCER_VERBS or 
        w in _MOTION_VERBS or w in _TRANSFER_VERBS or w in _POSITIONAL_VERBS):
        return "VERB"
    
    # Verb heuristics: common endings
    if w.endswith(("ting", "ning", "ding", "ring", "king", "ming",
                   "sing", "zing", "cing", "ping", "bing", "ving",
                   "ling", "ging")):
        return "VERB"  # present participle
    if w.endswith("ify") or w.endswith("ise") or w.endswith("ize"):
        return "VERB"
    if w.endswith("ates") or w.endswith("etes") or w.endswith("utes"):
        return "VERB"
    if w.endswith("ed") and len(w) > 3:
        return "VERB"  # past tense / past participle
    if w.endswith("es") and len(w) > 4 and not w.endswith("ness"):
        return "VERB"  # 3rd person singular
    if w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is", "ness", "ous", "ics")):
        # Could be plural noun OR 3rd-person verb; guess noun
        return "NOUN"
    # Adjective heuristics
    if any(w.endswith(suf) for suf in _ADJECTIVE_SUFFIXES):
        return "ADJ"
    # Default: treat as NOUN (content word)
    return "NOUN"


def _chunk_phrases(words: list) -> dict:
    """Chunk a word sequence into syntactic phrases.

    Returns a dict with keys: subject_np, verb, object_np, prep_phrases.
    Each NP is a dict: {det, adjs, head}.
    """
    tags = [_pos_tag_simple(w) for w in words]
    n = len(words)

    result = {
        "subject_np": None,
        "verb": None,
        "aux": None,
        "object_np": None,
        "prep_phrases": [],
    }

    i = 0

    # --- Parse subject NP ---
    np_result = _parse_np(words, tags, i)
    if np_result:
        result["subject_np"] = np_result["np"]
        i = np_result["end"]

    # --- Parse verb (possibly with aux) ---
    while i < n and tags[i] in ("ADV",):
        i += 1
    if i < n and tags[i] == "AUX":
        result["aux"] = words[i]
        i += 1
    if i < n and tags[i] in ("VERB", "NOUN", "ADJ"):
        # Accept NOUN here because our tagger may mis-tag verbs
        result["verb"] = words[i]
        i += 1
    elif i < n and tags[i] == "AUX" and result["aux"] is None:
        result["verb"] = words[i]
        i += 1
    
    # If we have aux but no main verb, treat aux as the verb (copula)
    if result["aux"] and not result["verb"]:
        result["verb"] = result["aux"]
        result["aux"] = None

    # --- Parse object NP ---
    np_result = _parse_np(words, tags, i)
    if np_result:
        result["object_np"] = np_result["np"]
        i = np_result["end"]

    # --- Parse PP* ---
    while i < n:
        if tags[i] == "PREP":
            prep = words[i]
            i += 1
            np_result = _parse_np(words, tags, i)
            if np_result:
                result["prep_phrases"].append({"prep": prep, "np": np_result["np"]})
                i = np_result["end"]
            else:
                break
        elif tags[i] == "CONJ":
            i += 1  # skip conjunctions
        else:
            i += 1  # skip unknown


    return result


def _parse_np(words: list, tags: list, start: int):
    """Try to parse an NP starting at position `start`.

    NP → (DET)? (ADJ|ADV)* (NOUN|PRON)+

    Returns None if no NP found, else {np: {det, adjs, head, head_words}, end: int}.
    """
    n = len(words)
    i = start

    det = None
    adjs = []
    head_words = []

    # Optional determiner
    if i < n and tags[i] == "DET":
        det = words[i]
        i += 1

    # Optional adjectives / adverbs
    while i < n and tags[i] in ("ADJ", "ADV"):
        adjs.append(words[i])
        i += 1

    # Head noun(s) — at least one NOUN or PRON needed
    while i < n and tags[i] in ("NOUN", "PRON"):
        head_words.append(words[i])
        i += 1

    # If no head nouns found but we have adjectives, treat last adj as head
    # (likely a mistagged proper noun or noun ending in -y, -ic, etc.)
    if not head_words and adjs:
        head_words = [adjs.pop()]
    
    if not head_words:
        return None

    return {
        "np": {"det": det, "adjs": adjs, "head": head_words[-1], "head_words": head_words},
        "end": i,
    }


def _encode_np(np_dict: dict) -> "hv.HyperVector":
    """Encode a Noun Phrase using VSA role-filler binding.

    NP_hv = bind(ROLE_det, det) ⊕ bind(ROLE_adj, adj_bundle) ⊕ bind(ROLE_head, head)
    """
    parts = []

    if np_dict.get("det"):
        det_hv = hv.HyperVector(_stable_seed(f"__kw__{np_dict['det']}"))
        parts.append(_role_hv("np_det").xor(det_hv))

    if np_dict.get("adjs"):
        adj_hvs = [hv.HyperVector(_stable_seed(f"__kw__{a}")) for a in np_dict["adjs"]]
        adj_bundle = _deterministic_bundle(adj_hvs)
        parts.append(_role_hv("np_adj").xor(adj_bundle))

    head_hv = hv.HyperVector(_stable_seed(f"__kw__{np_dict['head']}"))
    parts.append(_role_hv("np_head").xor(head_hv))

    # If multiple head words (compound noun), bundle them
    if len(np_dict.get("head_words", [])) > 1:
        hw_hvs = [hv.HyperVector(_stable_seed(f"__kw__{w}")) for w in np_dict["head_words"]]
        compound = _deterministic_bundle(hw_hvs)
        parts.append(_role_hv("np_compound").xor(compound))

    if not parts:
        return hv.HyperVector(0)

    return _deterministic_bundle(parts)


def _build_phrase_structure_hv(words: list):
    """Build a compositional phrase-structure HV from a word list.

    Uses the chunker to identify S → NP VP PP* structure, then assigns
    semantic roles (Agent, Patient, Theme, Instrument, etc.) based on
    verb semantics. Encodes with VSA role-filler binding.

    Returns None if the input is too short to parse.
    
    Example encoding for \"John broke the vase with a hammer\":
        sentence_hv = bundle(
            agent ⊗ NP(John),
            patient ⊗ NP(vase),
            instrument ⊗ NP(hammer),
            action ⊗ HV(broke)
        )
    """
    if len(words) < 2:
        return None

    parsed = _chunk_phrases(words)
    
    # NEW: Assign semantic roles based on verb class
    semantic_roles = _assign_semantic_roles(parsed)
    
    if not semantic_roles:
        return None
    
    tree_parts = []

    # Encode action (verb with auxiliary if present)
    if semantic_roles.get("action"):
        verb_hv = hv.HyperVector(_stable_seed(f"__kw__{semantic_roles['action']}"))
        if parsed.get("aux"):
            aux_hv = hv.HyperVector(_stable_seed(f"__kw__{parsed['aux']}"))
            verb_hv = _role_hv("aux").xor(aux_hv).xor(verb_hv)
        tree_parts.append(_role_hv("action").xor(verb_hv))
    
    # Encode semantic role fillers
    for role_name in ["agent", "patient", "experiencer", "theme", "subject", "object"]:
        if role_name in semantic_roles:
            filler_np = semantic_roles[role_name]
            filler_hv = _encode_np(filler_np)
            tree_parts.append(_role_hv(role_name).xor(filler_hv))
    
    # Encode spatial/instrumental roles from PP
    for role_name in ["instrument", "location", "source", "goal"]:
        if role_name in semantic_roles:
            filler_np = semantic_roles[role_name]
            filler_hv = _encode_np(filler_np)
            tree_parts.append(_role_hv(role_name).xor(filler_hv))
    
    # Encode generic modifiers (unclassified PPs)
    if "modifier" in semantic_roles:
        for idx, mod in enumerate(semantic_roles["modifier"]):
            prep_hv = hv.HyperVector(_stable_seed(f"__kw__{mod['prep']}"))
            np_hv = _encode_np(mod["np"])
            mod_hv = _role_hv("pp_prep").xor(prep_hv).xor(np_hv)
            # Permute by index to distinguish multiple modifiers
            if idx > 0:
                mod_hv = mod_hv.permute(idx)
            tree_parts.append(_role_hv("modifier").xor(mod_hv))

    if not tree_parts:
        return None

    return _deterministic_bundle(tree_parts)


# ---------------------------------------------------------------------------
# Universal Input
# ---------------------------------------------------------------------------
class UniversalInput:
    """
    Grounding module that converts heterogeneous data into 10 240-bit HVs.

    Usage::

        ui = UniversalInput()
        hv_scalar  = ui.ground(0.73, domain="sensor:temperature")
        hv_cat     = ui.ground("error",  domain="status")
        hv_dict    = ui.ground({"ip": "10.0.0.1", "status": "ok"}, domain="net")
        hv_seq     = ui.ground([0.1, 0.5, 0.9], domain="trajectory")
    """

    def __init__(
        self,
        n_bins: int = DEFAULT_THERMOMETER_BINS,
        max_codebook: int = MAX_CODEBOOK_SIZE,
    ):
        self.n_bins = n_bins
        self.max_codebook = max_codebook

        # LRU codebook:  "domain:label" → HyperVector
        self._codebook: collections.OrderedDict[str, hv.HyperVector] = (
            collections.OrderedDict()
        )

        # Thermometer bin HVs (lazy-init per domain)
        # domain → list[HyperVector]  (one per bin)
        self._bin_hvs: Dict[str, list] = {}

        # Role HVs for dict keys (lazy-init)
        self._role_hvs: Dict[str, hv.HyperVector] = {}

        # Stats for telemetry
        self._stats = {
            "scalars_grounded": 0,
            "categories_grounded": 0,
            "dicts_grounded": 0,
            "lists_grounded": 0,
            "codebook_size": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ground(
        self,
        data: Any,
        domain: str = "default",
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> hv.HyperVector:
        """Auto-detect type and ground *data* into a hypervector.

        Parameters
        ----------
        data : scalar | str | dict | list
        domain : namespacing tag (e.g. ``"sensor:temperature"``)
        min_val, max_val : range for scalar grounding (ignored for non-scalars)
        """
        if isinstance(data, dict):
            return self.ground_dict(data, domain)
        elif isinstance(data, (list, tuple)):
            return self.ground_sequence(data, domain, min_val, max_val)
        elif isinstance(data, str):
            # Use n-gram text grounding for multi-word strings
            if " " in data and len(data) > 5:
                return self.ground_text(data, domain)
            return self.ground_category(data, domain)
        elif isinstance(data, (int, float, np.integer, np.floating)):
            return self.ground_scalar(float(data), min_val, max_val, domain)
        else:
            # Fallback: hash repr as category
            return self.ground_category(repr(data), domain)

    # ---- Scalar -------------------------------------------------------
    def ground_scalar(
        self,
        value: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
        domain: str = "default",
    ) -> hv.HyperVector:
        """Thermometer encode *value* ∈ [min_val, max_val] into an HV.

        Nearby values produce highly similar vectors (Scalar Similarity Test).
        """
        bins = self._get_bin_hvs(domain)

        # Clamp + normalise to [0, 1]
        span = max_val - min_val
        if span == 0:
            t = 0.5
        else:
            t = (value - min_val) / span
        t = max(0.0, min(1.0, t))

        # Thermometer: activate bins 0 .. k  (k = floor(t * n_bins))
        k = int(t * (self.n_bins - 1))

        # Weighted bundle of the active bins (closer bins get full weight,
        # further bins taper linearly).
        # For efficiency we XOR-bundle neighbours in a window ±3 around k.
        window = 3
        lo = max(0, k - window)
        hi = min(self.n_bins - 1, k + window)

        window_bins = [bins[i] for i in range(lo, hi + 1)]
        # Deterministic bundle ensures identical scalars yield identical HVs
        result = _deterministic_bundle(window_bins)

        self._stats["scalars_grounded"] += 1
        return result

    # ---- Category -----------------------------------------------------
    def ground_category(self, label: str, domain: str = "default") -> hv.HyperVector:
        """Deterministic HV for a categorical label (with LRU eviction)."""
        key = f"{domain}:{label}"

        if key in self._codebook:
            # Move to end (most recent)
            self._codebook.move_to_end(key)
            self._stats["categories_grounded"] += 1
            return self._codebook[key]

        # Generate new HV from deterministic seed
        seed = _stable_seed(key)
        vec = hv.HyperVector(seed)

        # LRU eviction
        if len(self._codebook) >= self.max_codebook:
            self._codebook.popitem(last=False)

        self._codebook[key] = vec
        self._stats["categories_grounded"] += 1
        self._stats["codebook_size"] = len(self._codebook)
        return vec

    # ---- Stopwords for keyword extraction ---------------------------------
    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "and", "but", "or",
        "nor", "not", "so", "very", "just", "than", "too", "also", "that",
        "this", "these", "those", "it", "its", "they", "them", "their",
        "we", "our", "you", "your", "he", "she", "him", "her", "his",
        "my", "me", "i", "if", "no", "up", "about",
    })

    # ---- Text (semantic keyword + n-gram + phrase-structure enhanced) -----
    def ground_text(self, text: str, domain: str = "default", ngram_size: int = 3) -> hv.HyperVector:
        """Ground natural-language text using **4-component architecture**.

        Pipeline
        --------
        1. Normalise (lowercase, strip punctuation).
        2. **Keyword HV** — topical similarity via shared content words.
        3. **Character n-gram HV** — sub-word / morphological overlap.
        4. **Word-order HV** — positional permutation encoding.
        5. **Phrase-structure HV** — compositional parse tree encoded with
           VSA role-filler binding (Subject ⊗ NP, Predicate ⊗ VP, etc.).
        6. Combine all four via segment concatenation (50/15/15/20).
        """
        import re

        # --- normalise ----
        clean = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
        if not clean:
            return self.ground_category(text, domain)

        words = clean.split()

        # ─── Component 1: KEYWORD (semantic-topical) HV ──────────────
        keywords = [w for w in words if w not in self._STOP_WORDS and len(w) > 2]
        if not keywords:
            keywords = [w for w in words if len(w) > 1]  # fallback

        if keywords:
            kw_hvs = []
            for kw in keywords:
                seed = _stable_seed(f"__kw__{kw}")  # domain-independent!
                kw_hvs.append(hv.HyperVector(seed))
            keyword_hv = _deterministic_bundle(kw_hvs)
        else:
            keyword_hv = None

        # ─── Component 2: CHARACTER N-GRAM HV ────────────────────────
        padded = f" {clean} "
        ngrams: list[str] = []
        for i in range(len(padded) - ngram_size + 1):
            ngrams.append(padded[i : i + ngram_size])

        ngram_hv = None
        if ngrams:
            ngram_counts: dict[str, int] = {}
            for ng in ngrams:
                ngram_counts[ng] = ngram_counts.get(ng, 0) + 1

            ng_hvs: list[hv.HyperVector] = []
            for ng in ngram_counts:
                seed = _stable_seed(f"__ngram__{domain}:{ng}")
                ng_hvs.append(hv.HyperVector(seed))

            ngram_hv = _deterministic_bundle(ng_hvs)

        # ─── Component 3: WORD-ORDER HV ──────────────────────────────
        order_hv = None
        first_words = words[:10]
        if len(first_words) > 1:
            w_hvs: list[hv.HyperVector] = []
            for i, w in enumerate(first_words):
                w_hv = hv.HyperVector(_stable_seed(f"__word__{domain}:{w}"))
                if i > 0:
                    w_hv = w_hv.permute(i)
                w_hvs.append(w_hv)
            order_hv = _deterministic_bundle(w_hvs)

        # ─── Component 4: PHRASE-STRUCTURE HV ────────────────────────
        phrase_hv = _build_phrase_structure_hv(words)

        # ─── Combine: segment-based concatenation ────────────────────
        #   bits [0,        kw_end)  ← keyword HV    (50%)  topical sim
        #   bits [kw_end,   ng_end)  ← n-gram HV     (15%)  sub-word sim
        #   bits [ng_end,   or_end)  ← order HV      (15%)  word-order sim
        #   bits [or_end,   D)       ← phrase HV     (20%)  compositional
        D = DIMENSION
        kw_end = int(D * 0.50)   # 5120
        ng_end = kw_end + int(D * 0.15)  # 6656
        or_end = ng_end + int(D * 0.15)  # 8192

        result_bits = np.zeros(D, dtype=np.int8)
        filler = hv.HyperVector(0)  # fallback random bits

        if keyword_hv is not None:
            result_bits[:kw_end] = keyword_hv.bits[:kw_end]
        else:
            result_bits[:kw_end] = filler.bits[:kw_end]

        if ngram_hv is not None:
            result_bits[kw_end:ng_end] = ngram_hv.bits[kw_end:ng_end]
        else:
            result_bits[kw_end:ng_end] = filler.bits[kw_end:ng_end]

        if order_hv is not None:
            result_bits[ng_end:or_end] = order_hv.bits[ng_end:or_end]
        else:
            result_bits[ng_end:or_end] = filler.bits[ng_end:or_end]

        if phrase_hv is not None:
            result_bits[or_end:] = phrase_hv.bits[or_end:]
        else:
            result_bits[or_end:] = filler.bits[or_end:]

        result = hv.HyperVector.from_bits(result_bits)

        self._stats["categories_grounded"] += 1
        self._stats["codebook_size"] = len(self._codebook)
        return result

    # ---- Dict (recursive role-filler binding) -------------------------
    def ground_dict(self, data: dict, domain: str = "default") -> hv.HyperVector:
        """Bind every (key, value) pair with role-HV, then XOR-bundle."""
        if not data:
            return hv.HyperVector(0)

        parts = []
        for key, val in data.items():
            role_hv = self._get_role_hv(key, domain)
            # Recursively ground the value
            val_hv = self.ground(val, domain=f"{domain}.{key}")
            # Bind role ⊗ filler
            bound = role_hv.xor(val_hv)
            parts.append(bound)

        # XOR-bundle all pairs
        result = parts[0]
        for p in parts[1:]:
            result = result.xor(p)

        self._stats["dicts_grounded"] += 1
        return result

    # ---- List (sequence encoding) -------------------------------------
    def ground_sequence(
        self,
        data: list,
        domain: str = "default",
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> hv.HyperVector:
        """Encode an ordered sequence using permutation: Σ ρ^i(item_i).

        Preserves order: [A, B, C] ≠ [B, A, C].
        """
        if not data:
            return hv.HyperVector(0)

        parts = []
        for i, item in enumerate(data):
            item_hv = self.ground(item, domain=domain, min_val=min_val, max_val=max_val)
            # Apply i-th permutation for positional encoding
            if i > 0:
                item_hv = item_hv.permute(i)
            parts.append(item_hv)

        # XOR-bundle all position-encoded items
        result = parts[0]
        for p in parts[1:]:
            result = result.xor(p)

        self._stats["lists_grounded"] += 1
        return result

    # ------------------------------------------------------------------
    # Introspection (for dashboard telemetry)
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Return grounding statistics for dashboard display."""
        self._stats["codebook_size"] = len(self._codebook)
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_bin_hvs(self, domain: str) -> list:
        """Lazily generate n_bins base HVs for thermometer encoding."""
        if domain not in self._bin_hvs:
            base_seed = _stable_seed(f"__thermo__{domain}")
            bins = []
            for i in range(self.n_bins):
                bins.append(hv.HyperVector((base_seed + i) & 0xFFFF_FFFF))
            self._bin_hvs[domain] = bins
        return self._bin_hvs[domain]

    def _get_role_hv(self, key: str, domain: str) -> hv.HyperVector:
        """Lazily generate a role HV for a dict key."""
        full_key = f"__role__{domain}:{key}"
        if full_key not in self._role_hvs:
            self._role_hvs[full_key] = hv.HyperVector(_stable_seed(full_key))
        return self._role_hvs[full_key]
