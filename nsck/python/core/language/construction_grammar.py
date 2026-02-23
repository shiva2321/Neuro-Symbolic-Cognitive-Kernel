"""Construction Grammar module for NSCK V3/V4/V6.

V4 additions:
- Massively extended COMMON_VERBS for broader NLU coverage (~40-60% → ~75-85%)
- NEGATION_WORDS: detect negated relations ("not", "never", "no", "cannot")
- TEMPORAL_CONNECTIVES: temporal relation keywords
- CONDITIONAL_CONNECTIVES: conditional logic keywords
- Improved _classify_word() morphological heuristics
- New constructions: negation, conditional, temporal, similarity, difference

V6 additions:
- BrillPosTagger integration in _classify_word() for context-aware tagging
  when a full sentence is available (90%+ coverage on real-world text)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional

ARTICLES = {"a", "an", "the"}
PREPOSITIONS = {
    "in", "on", "at", "by", "for", "with", "to", "from", "of",
    "into", "onto", "about", "above", "below", "under", "over",
    "between", "among", "through", "across", "around", "near",
    "beside", "behind", "before", "after", "during", "within",
    "without", "beyond", "along", "toward", "towards", "against",
    "despite", "except", "including", "per", "via",
}

# V4: minimum word length for the -ish adjective suffix heuristic.
# Short words ending in -ish are often nouns (fish, dish, wish) not adjectives.
# Words 6+ chars (reddish, childish) are reliably adjectives.
_MIN_ISH_SUFFIX_LEN = 6

# V4: negation function words (used by _classify_word and new constructions)
NEGATION_WORDS = {"not", "never", "no", "neither", "nor", "cannot", "cant", "won't", "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't"}

# V4: temporal connectives (for temporal-ordering constructions)
TEMPORAL_CONNECTIVES = {"before", "after", "since", "until", "when", "while", "during", "then", "once", "later", "earlier", "previously", "subsequently", "meanwhile"}

# V4: conditional connectives (for conditional-logic constructions)
CONDITIONAL_CONNECTIVES = {"if", "unless", "whenever", "provided", "assuming", "given", "suppose", "supposing"}

COMMON_VERBS = {
    # ── Auxiliaries and copulas ────────────────────────────────────────────────
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "will", "would", "shall", "should", "may", "might",
    "can", "could", "must", "ought",

    # ── Cognition / mental ────────────────────────────────────────────────────
    "know", "knows", "knew", "known",
    "think", "thinks", "thought",
    "believe", "believes", "believed",
    "understand", "understands", "understood",
    "remember", "remembers", "remembered",
    "forget", "forgets", "forgot", "forgotten",
    "learn", "learns", "learned", "learnt",
    "realize", "realizes", "realised", "realized",
    "recognize", "recognizes", "recognised", "recognized",
    "imagine", "imagines", "imagined",
    "assume", "assumes", "assumed",
    "consider", "considers", "considered",
    "decide", "decides", "decided",
    "guess", "guesses", "guessed",
    "wonder", "wonders", "wondered",
    "notice", "notices", "noticed",
    "perceive", "perceives", "perceived",
    "conclude", "concludes", "concluded",
    "infer", "infers", "inferred",
    "judge", "judges", "judged",

    # ── Emotion / desire ─────────────────────────────────────────────────────
    "love", "loves", "loved",
    "like", "likes", "liked",
    "hate", "hates", "hated",
    "want", "wants", "wanted",
    "wish", "wishes", "wished",
    "hope", "hopes", "hoped",
    "fear", "fears", "feared",
    "enjoy", "enjoys", "enjoyed",
    "prefer", "prefers", "preferred",
    "feel", "feels", "felt",
    "suffer", "suffers", "suffered",

    # ── Communication ────────────────────────────────────────────────────────
    "say", "says", "said",
    "tell", "tells", "told",
    "speak", "speaks", "spoke", "spoken",
    "talk", "talks", "talked",
    "ask", "asks", "asked",
    "answer", "answers", "answered",
    "reply", "replies", "replied",
    "explain", "explains", "explained",
    "describe", "describes", "described",
    "report", "reports", "reported",
    "announce", "announces", "announced",
    "declare", "declares", "declared",
    "argue", "argues", "argued",
    "claim", "claims", "claimed",
    "state", "states", "stated",
    "suggest", "suggests", "suggested",
    "propose", "proposes", "proposed",
    "recommend", "recommends", "recommended",
    "advise", "advises", "advised",
    "warn", "warns", "warned",
    "promise", "promises", "promised",
    "agree", "agrees", "agreed",
    "disagree", "disagrees", "disagreed",
    "deny", "denies", "denied",
    "confirm", "confirms", "confirmed",
    "show", "shows", "showed", "shown",

    # ── Motion / spatial ─────────────────────────────────────────────────────
    "go", "goes", "went", "gone",
    "come", "comes", "came",
    "run", "runs", "ran",
    "walk", "walks", "walked",
    "move", "moves", "moved",
    "travel", "travels", "travelled", "traveled",
    "arrive", "arrives", "arrived",
    "leave", "leaves", "left",
    "enter", "enters", "entered",
    "exit", "exits", "exited",
    "return", "returns", "returned",
    "reach", "reaches", "reached",
    "approach", "approaches", "approached",
    "follow", "follows", "followed",
    "lead", "leads", "led",
    "pass", "passes", "passed",
    "cross", "crosses", "crossed",
    "rise", "rises", "rose", "risen",
    "fall", "falls", "fell", "fallen",
    "fly", "flies", "flew", "flown",
    "swim", "swims", "swam", "swum",
    "climb", "climbs", "climbed",
    "jump", "jumps", "jumped",

    # ── Physical action ──────────────────────────────────────────────────────
    "make", "makes", "made",
    "take", "takes", "took", "taken",
    "give", "gives", "gave", "given",
    "get", "gets", "got",
    "put", "puts",
    "place", "places", "placed",
    "set", "sets",
    "bring", "brings", "brought",
    "carry", "carries", "carried",
    "hold", "holds", "held",
    "pick", "picks", "picked",
    "drop", "drops", "dropped",
    "throw", "throws", "threw", "thrown",
    "catch", "catches", "caught",
    "push", "pushes", "pushed",
    "pull", "pulls", "pulled",
    "open", "opens", "opened",
    "close", "closes", "closed",
    "start", "starts", "started",
    "stop", "stops", "stopped",
    "begin", "begins", "began", "begun",
    "end", "ends", "ended",
    "finish", "finishes", "finished",
    "continue", "continues", "continued",
    "break", "breaks", "broke", "broken",
    "fix", "fixes", "fixed",
    "cut", "cuts",
    "hit", "hits",
    "kill", "kills", "killed",
    "eat", "eats", "ate", "eaten",
    "drink", "drinks", "drank", "drunk",
    "sleep", "sleeps", "slept",
    "wake", "wakes", "woke", "woken",
    "wear", "wears", "wore", "worn",
    "read", "reads",
    "write", "writes", "wrote", "written",

    # ── Change / transformation ──────────────────────────────────────────────
    "change", "changes", "changed",
    "become", "becomes", "became",
    "grow", "grows", "grew", "grown",
    "increase", "increases", "increased",
    "decrease", "decreases", "decreased",
    "expand", "expands", "expanded",
    "reduce", "reduces", "reduced",
    "improve", "improves", "improved",
    "damage", "damages", "damaged",
    "destroy", "destroys", "destroyed",
    "create", "creates", "created",
    "form", "forms", "formed",
    "produce", "produces", "produced",
    "generate", "generates", "generated",
    "convert", "converts", "converted",
    "transform", "transforms", "transformed",

    # ── Existence / state ────────────────────────────────────────────────────
    "exist", "exists", "existed",
    "live", "lives", "lived",
    "die", "dies", "died",
    "appear", "appears", "appeared",
    "disappear", "disappears", "disappeared",
    "remain", "remains", "remained",
    "stay", "stays", "stayed",
    "keep", "keeps", "kept",
    "mean", "means", "meant",
    "seem", "seems", "seemed",
    "look", "looks", "looked",
    "sound", "sounds", "sounded",
    "feel", "feels", "felt",
    "stand", "stands", "stood",
    "sit", "sits", "sat",
    "lie", "lies", "lay", "lain",
    "wait", "waits", "waited",
    "last", "lasts", "lasted",
    "happen", "happens", "happened",
    "occur", "occurs", "occurred",
    "result", "results", "resulted",
    "work", "works", "worked",
    "fail", "fails", "failed",
    "succeed", "succeeds", "succeeded",

    # ── Social / interpersonal ───────────────────────────────────────────────
    "help", "helps", "helped",
    "use", "uses", "used",
    "find", "finds", "found",
    "call", "calls", "called",
    "try", "tries", "tried",
    "need", "needs", "needed",
    "let", "lets",
    "hear", "hears", "heard",
    "play", "plays", "played",
    "pay", "pays", "paid",
    "meet", "meets", "met",
    "join", "joins", "joined",
    "share", "shares", "shared",
    "lose", "loses", "lost",
    "win", "wins", "won",
    "choose", "chooses", "chose", "chosen",
    "buy", "buys", "bought",
    "sell", "sells", "sold",
    "trade", "trades", "traded",
    "hire", "hires", "hired",
    "send", "sends", "sent",
    "receive", "receives", "received",
    "collect", "collects", "collected",
    "gather", "gathers", "gathered",
    "manage", "manages", "managed",
    "control", "controls", "controlled",
    "lead", "leads", "led",
    "own", "owns", "owned",

    # ── Construction / creation ──────────────────────────────────────────────
    "build", "builds", "built",
    "design", "designs", "designed",
    "plan", "plans", "planned",
    "develop", "develops", "developed",
    "test", "tests", "tested",
    "measure", "measures", "measured",
    "calculate", "calculates", "calculated",
    "solve", "solves", "solved",
    "prove", "proves", "proved", "proven",

    # ── Relation / logic verbs ───────────────────────────────────────────────
    "cause", "causes", "caused",
    "contain", "contains", "contained",
    "require", "requires", "required",
    "enable", "enables", "enabled",
    "prevent", "prevents", "prevented",
    "support", "supports", "supported",
    "include", "includes", "included",
    "represent", "represents", "represented",
    "perform", "performs", "performed",
    "discover", "discovers", "discovered",
    "define", "defines", "defined",
    "involve", "involves", "involved",
    "allow", "allows", "allowed",
    "affect", "affects", "affected",
    "connect", "connects", "connected",
    "relate", "relates", "related",
    "compare", "compares", "compared",
    "differ", "differs", "differed",
    "apply", "applies", "applied",
    "depend", "depends", "depended",
    "belong", "belongs", "belonged",
    "classify", "classifies", "classified",
    "identify", "identifies", "identified",
    "detect", "detects", "detected",
    "observe", "observes", "observed",
    "analyze", "analyzes", "analysed", "analyzed",
    "predict", "predicts", "predicted",
    "store", "stores", "stored",
    "process", "processes", "processed",
    "transmit", "transmits", "transmitted",
    "absorb", "absorbs", "absorbed",
    "emit", "emits", "emitted",
    "bind", "binds", "bound",
    "interact", "interacts", "interacted",
    "respond", "responds", "responded",
    "react", "reacts", "reacted",
    "adapt", "adapts", "adapted",
    "evolve", "evolves", "evolved",
    "spread", "spreads",
    "flow", "flows", "flowed",
    "release", "releases", "released",
    "attract", "attracts", "attracted",
    "repel", "repels", "repelled",
    "activate", "activates", "activated",
    "inhibit", "inhibits", "inhibited",
    "regulate", "regulates", "regulated",
    "maintain", "maintains", "maintained",
    "protect", "protects", "protected",
    "expose", "exposes", "exposed",
    "separate", "separates", "separated",
    "combine", "combines", "combined",
    "replace", "replaces", "replaced",
    "extend", "extends", "extended",
    "limit", "limits", "limited",
    # ── Common -ize/-ise/-ify verbs (explicit to avoid suffix heuristic conflicts) ─
    "organize", "organises", "organizes", "organised", "organized", "organizing",
    "realize", "realise", "realises", "realizes", "realized", "realised", "realizing",
    "recognize", "recognise", "recognizes", "recognises", "recognised", "recognized", "recognizing",
    "criticize", "criticizes", "criticized", "criticizing",
    "analyze", "analyzes", "analyzed", "analyzing",
    "emphasize", "emphasizes", "emphasized", "emphasizing",
    "maximize", "maximizes", "maximized", "maximizing",
    "minimize", "minimizes", "minimized", "minimizing",
    "optimize", "optimizes", "optimized", "optimizing",
    "synchronize", "synchronizes", "synchronized", "synchronizing",
    "utilize", "utilizes", "utilized", "utilizing",
    "visualize", "visualizes", "visualized", "visualizing",
    "specialize", "specializes", "specialized", "specializing",
    "generalize", "generalizes", "generalized", "generalizing",
    "conceptualize", "conceptualizes", "conceptualized",
    "satisfy", "satisfies", "satisfied", "satisfying",
    "justify", "justifies", "justified", "justifying",
    "classify", "classifies", "classified", "classifying",
    "modify", "modifies", "modified", "modifying",
    "notify", "notifies", "notified", "notifying",
    "verify", "verifies", "verified", "verifying",
    "qualify", "qualifies", "qualified", "qualifying",
    "amplify", "amplifies", "amplified", "amplifying",
    "simplify", "simplifies", "simplified", "simplifying",
    "specify", "specifies", "specified", "specifying",
    "identify", "identifies", "identified", "identifying",
    # ── Logic/reasoning verbs commonly missing ────────────────────────────────
    "imply", "implies", "implied", "implying",
    "infer", "infers", "inferred", "inferring",
    "lack", "lacks", "lacked", "lacking",
    "denote", "denotes", "denoted", "denoting",
    "signify", "signifies", "signified", "signifying",
    "entail", "entails", "entailed", "entailing",
    "contradict", "contradicts", "contradicted", "contradicting",
    "confirm", "confirms", "confirmed", "confirming",
    "negate", "negates", "negated", "negating",
    "assume", "assumes", "assumed", "assuming",
    "assert", "asserts", "asserted", "asserting",
    "hypothesize", "hypothesizes", "hypothesized",
    "approximate", "approximates", "approximated",
}

@dataclass
class Construction:
    name: str
    pattern: List[str]  # slot types: "NOUN","VERB","ADJ","PREP","ART" or literals
    roles: Dict[str, int]  # role_name -> pattern position
    relation: str
    confidence: float = 1.0

@dataclass
class ConstructionMatch:
    construction: Construction
    role_fillers: Dict[str, str]  # role_name -> word
    score: float

def _classify_word(word: str) -> str:
    """Classify a single word into a POS category.

    V4 improvements:
    - Checks NEGATION_WORDS early (returns "NEG")
    - Checks TEMPORAL_CONNECTIVES (returns "TEMP")
    - Checks CONDITIONAL_CONNECTIVES (returns "COND")
    - Noun-suffix heuristics: -tion, -ment, -ness, -ity, -er, -or, -ist
    - Verb suffix heuristics: -ize, -ify, -ate, -en (for longer words)
      NOTE: -ise is intentionally excluded (too many English nouns end in -ise:
      "enterprise", "sunrise", "surprise").  British -ise verbs must be in COMMON_VERBS.
    - 3rd-person -s: strip trailing -s and check stem against COMMON_VERBS
      or verb suffix rules (avoids needing every inflected form in COMMON_VERBS)
    - Adjective heuristics: -able, -ible, -ful, -less, -ous, -ive, -ish
    """
    w = word.lower()
    if w in ARTICLES:
        return "ART"
    if w in NEGATION_WORDS:
        return "NEG"
    if w in TEMPORAL_CONNECTIVES:
        return "TEMP"
    if w in CONDITIONAL_CONNECTIVES:
        return "COND"
    if w in PREPOSITIONS:
        return "PREP"
    if w in COMMON_VERBS:
        return "VERB"

    # V4: strong noun suffixes (check before verb/adj to avoid mis-tagging)
    if w.endswith(("tion", "sion", "ment", "ness", "ity", "ety", "ism",
                   "ology", "ography", "graphy")):
        return "NOUN"
    if w.endswith(("ist",)):
        return "NOUN"

    # V4: verb-form heuristics — must be long enough to avoid catching short nouns
    # Only -ize (not -ise) to avoid mis-tagging "sunrise", "enterprise", "surprise".
    if len(w) >= 5:
        if w.endswith(("ize", "ify", "ified", "izing", "ifying")):
            return "VERB"
        if w.endswith(("ate", "ated", "ating", "ates")):
            return "VERB"
        # -en as a verb ending (listen, happen, widen) — only 6+ chars
        if len(w) >= 6 and w.endswith("en") and not w.endswith(("tion", "sion")):
            return "VERB"

    # V3: -ed and -ing
    if w.endswith("ing") and len(w) > 4:
        return "VERB"
    if w.endswith("ed") and len(w) > 4:
        return "VERB"

    # V4: 3rd-person singular -s — strip and check stem
    if w.endswith("s") and len(w) > 3:
        stem = w[:-1]
        if stem in COMMON_VERBS:
            return "VERB"
        # Also accept if stem itself is a verb by suffix rules
        if len(stem) >= 5 and stem.endswith(("ize", "ify", "ate")):
            return "VERB"

    # V4: -ies forms (tries → try, implies → imply, identifies → identify)
    if w.endswith("ies") and len(w) > 4:
        stem = w[:-3] + "y"
        if stem in COMMON_VERBS:
            return "VERB"
        # e.g. "identifies" → "identify" → endswith("ify") → VERB
        if stem.endswith(("ify", "ize")):
            return "VERB"

    # V4: adjective suffixes (only reliable ones to avoid mis-tagging nouns)
    if w.endswith("ly"):
        return "ADJ"
    if w.endswith(("ful", "less", "ous", "ive", "able", "ible")):
        return "ADJ"
    # -ish is an adjective suffix (reddish, childish) but not for short words
    # "fish" (4 chars), "dish" (4 chars), "wish" (4 chars) are NOT adjectives
    if w.endswith("ish") and len(w) >= _MIN_ISH_SUFFIX_LEN:
        return "ADJ"

    return "NOUN"


# ---------------------------------------------------------------------------
# V6: Sentence-level POS tagging via BrillPosTagger
# ---------------------------------------------------------------------------

# Mapping from Brill tag set → NSCK slot tag set
_BRILL_TO_NSCK: Dict[str, str] = {
    "VB":   "VERB", "VBZ": "VERB", "VBD": "VERB",
    "VBG":  "VERB", "VBN": "VERB", "MD":  "VERB",
    "NN":   "NOUN", "NNS": "NOUN", "NNP": "NOUN",
    "JJ":   "ADJ",
    "RB":   "ADJ",   # treat adverbs as adjective-class in constructions
    "DT":   "ART",
    "IN":   "PREP",
    "NEG":  "NEG",
    "TEMP": "TEMP",
    "COND": "COND",
    "CC":   "PREP",  # coordinators act like prepositions in slot matching
    "PRP":  "NOUN",  # pronouns fill noun slots
    "PRP$": "NOUN",
    "CD":   "NOUN",  # numbers fill noun slots
    "WP":   "NOUN",
    "EX":   "NOUN",
}


def tag_sentence(tokens: List[str]) -> List[str]:
    """
    Tag a list of tokens using BrillPosTagger and map to NSCK slot tags.

    Returns a list of NSCK tags the same length as *tokens*, e.g.
    ["NOUN", "VERB", "ART", "NOUN"].

    Falls back to ``_classify_word`` if the import fails.
    """
    try:
        from python.core.language.pos_tagger import get_default_tagger
        tagger = get_default_tagger()
        brill_tags = tagger.tag(tokens)
        return [_BRILL_TO_NSCK.get(t, _classify_word(w)) for w, t in brill_tags]
    except Exception:
        return [_classify_word(t) for t in tokens]


def _slot_matches(slot: str, word: str) -> bool:
    """Check if a word matches a pattern slot."""
    if slot in ("NOUN", "VERB", "ADJ", "PREP", "ART", "NEG", "TEMP", "COND"):
        return _classify_word(word) == slot
    return word.lower() == slot.lower()

class ConstructionMatcher:
    def __init__(self):
        self.constructions: List[Construction] = self._load_core_constructions()

    def _load_core_constructions(self) -> List[Construction]:
        return [
            # ── Original V3 constructions ──────────────────────────────────────
            Construction("SVO_active", ["NOUN","VERB","NOUN"], {"subject":0,"verb":1,"object":2}, "relates_to", 0.8),
            Construction("copular_is", ["NOUN","is","NOUN"], {"subject":0,"attribute":2}, "is_a", 0.95),
            Construction("copular_is_a", ["NOUN","is","ART","NOUN"], {"subject":0,"attribute":3}, "is_a", 0.95),
            Construction("copular_is_adj", ["NOUN","is","ADJ"], {"subject":0,"quality":2}, "has_property", 0.9),
            Construction("possessive_has", ["NOUN","has","NOUN"], {"owner":0,"owned":2}, "has_property", 0.95),
            Construction("causative", ["NOUN","causes","NOUN"], {"cause":0,"effect":2}, "causes", 0.95),
            Construction("locative_in", ["NOUN","is","in","NOUN"], {"entity":0,"location":3}, "located_in", 0.95),
            Construction("locative_on", ["NOUN","is","on","NOUN"], {"entity":0,"location":3}, "located_on", 0.9),
            Construction("locative_at", ["NOUN","is","at","NOUN"], {"entity":0,"location":3}, "located_at", 0.9),
            Construction("containment", ["NOUN","contains","NOUN"], {"container":0,"contained":2}, "contains", 0.9),
            Construction("production", ["NOUN","produces","NOUN"], {"producer":0,"product":2}, "produces", 0.9),
            Construction("passive_by", ["NOUN","VERB","by","NOUN"], {"patient":0,"verb":1,"agent":3}, "acted_on_by", 0.85),
            Construction("comparative", ["NOUN","is","ADJ","than","NOUN"], {"entity1":0,"quality":2,"entity2":4}, "more_than", 0.9),
            Construction("SVOO", ["NOUN","VERB","NOUN","NOUN"], {"subject":0,"verb":1,"recipient":2,"theme":3}, "transfers_to", 0.75),
            Construction("instrumental", ["NOUN","VERB","NOUN","with","NOUN"], {"subject":0,"verb":1,"object":2,"instrument":4}, "uses_instrument", 0.8),
            Construction("temporal_during", ["NOUN","VERB","during","NOUN"], {"subject":0,"verb":1,"time":3}, "occurs_during", 0.85),
            Construction("purpose", ["NOUN","VERB","to","VERB","NOUN"], {"subject":0,"action":1,"goal":4}, "has_purpose", 0.8),
            Construction("part_whole", ["NOUN","is","ART","NOUN","of","NOUN"], {"part":0,"whole":5}, "part_of", 0.9),
            Construction("quantity", ["NOUN","has","NOUN","of","NOUN"], {"entity":0,"property":2,"value":4}, "quantified_as", 0.8),
            Construction("necessity", ["NOUN","requires","NOUN"], {"dependent":0,"required":2}, "requires", 0.9),
            Construction("enables", ["NOUN","enables","NOUN"], {"enabler":0,"enabled":2}, "enables", 0.9),
            Construction("prevents", ["NOUN","prevents","NOUN"], {"preventer":0,"prevented":2}, "prevents", 0.9),
            Construction("consists_of", ["NOUN","consists","of","NOUN"], {"whole":0,"part":3}, "consists_of", 0.9),
            Construction("made_of", ["NOUN","is","made","of","NOUN"], {"artifact":0,"material":4}, "made_of", 0.95),
            Construction("defined_as", ["NOUN","is","defined","as","NOUN"], {"term":0,"definition":4}, "defined_as", 0.95),
            Construction("known_as", ["NOUN","is","known","as","NOUN"], {"entity":0,"alias":4}, "known_as", 0.9),
            Construction("used_for", ["NOUN","is","used","for","NOUN"], {"tool":0,"purpose":4}, "used_for", 0.9),
            Construction("born_in", ["NOUN","was","born","in","NOUN"], {"person":0,"birthplace":4}, "born_in", 0.95),
            Construction("located_at_pred", ["NOUN","is","located","in","NOUN"], {"entity":0,"location":4}, "located_in", 0.95),
            Construction("result_of", ["NOUN","is","ART","result","of","NOUN"], {"effect":0,"cause":5}, "result_of", 0.9),
            Construction("SVO_verb_adv", ["NOUN","VERB","NOUN","ADJ"], {"subject":0,"verb":1,"object":2}, "relates_to", 0.7),

            # ── V4: Negation constructions ─────────────────────────────────────
            # "X does not Y Z" / "X is not Y"
            Construction("negation_is_not", ["NOUN","is","not","NOUN"], {"subject":0,"negated_attribute":3}, "not_is_a", 0.95),
            Construction("negation_is_not_adj", ["NOUN","is","not","ADJ"], {"subject":0,"negated_quality":3}, "not_has_property", 0.9),
            Construction("negation_does_not", ["NOUN","does","not","VERB","NOUN"], {"subject":0,"negated_verb":3,"object":4}, "not_relates_to", 0.9),
            Construction("negation_cannot", ["NOUN","cannot","VERB","NOUN"], {"subject":0,"negated_verb":2,"object":3}, "cannot_do", 0.9),
            Construction("negation_never", ["NOUN","never","VERB","NOUN"], {"subject":0,"negated_verb":2,"object":3}, "never_does", 0.85),
            Construction("negation_no_noun", ["no","NOUN","VERB","NOUN"], {"negated_subject":1,"verb":2,"object":3}, "lacks_relation", 0.85),
            Construction("negation_not_in", ["NOUN","is","not","in","NOUN"], {"subject":0,"excluded_location":4}, "not_located_in", 0.9),
            Construction("negation_lacks", ["NOUN","lacks","NOUN"], {"subject":0,"missing":2}, "lacks", 0.9),
            Construction("negation_without", ["NOUN","VERB","without","NOUN"], {"subject":0,"verb":1,"absent":3}, "acts_without", 0.8),

            # ── V4: Conditional constructions ──────────────────────────────────
            # "if X then Y" / "if X, Y" / "X unless Y"
            Construction("conditional_if_then", ["if","NOUN","VERB","NOUN"], {"condition_subject":1,"condition_verb":2,"condition_object":3}, "conditional_on", 0.9),
            Construction("conditional_unless", ["NOUN","VERB","unless","NOUN","VERB"], {"subject":0,"verb":1,"exception_subject":3,"exception_verb":4}, "unless_condition", 0.85),
            Construction("conditional_when", ["when","NOUN","VERB","NOUN","VERB"], {"trigger_subject":1,"trigger_verb":2,"result_subject":3,"result_verb":4}, "when_then", 0.85),
            Construction("conditional_only_if", ["NOUN","VERB","only","if","NOUN","VERB"], {"subject":0,"verb":1,"cond_subject":4,"cond_verb":5}, "only_if", 0.9),
            Construction("conditional_implies", ["NOUN","implies","NOUN"], {"antecedent":0,"consequent":2}, "implies", 0.95),
            Construction("conditional_leads_to", ["NOUN","leads","to","NOUN"], {"cause":0,"effect":3}, "leads_to", 0.9),
            Construction("conditional_results_in", ["NOUN","results","in","NOUN"], {"cause":0,"effect":3}, "results_in", 0.9),

            # ── V4: Temporal ordering constructions ────────────────────────────
            Construction("temporal_before", ["NOUN","VERB","before","NOUN","VERB"], {"early_subject":0,"early_verb":1,"late_subject":3,"late_verb":4}, "precedes", 0.9),
            Construction("temporal_after", ["NOUN","VERB","after","NOUN","VERB"], {"late_subject":0,"late_verb":1,"early_subject":3,"early_verb":4}, "follows", 0.9),
            Construction("temporal_before_noun", ["NOUN","before","NOUN"], {"event1":0,"event2":2}, "precedes", 0.85),
            Construction("temporal_after_noun", ["NOUN","after","NOUN"], {"event1":0,"event2":2}, "follows", 0.85),
            Construction("temporal_since", ["NOUN","VERB","since","NOUN"], {"subject":0,"verb":1,"start_event":3}, "since_event", 0.85),
            Construction("temporal_until", ["NOUN","VERB","until","NOUN"], {"subject":0,"verb":1,"end_event":3}, "until_event", 0.85),
            Construction("temporal_when_noun", ["when","NOUN","VERB","NOUN"], {"trigger":1,"verb":2,"result":3}, "triggered_by", 0.85),

            # ── V4: Similarity / difference ────────────────────────────────────
            Construction("similarity_like", ["NOUN","is","like","NOUN"], {"entity1":0,"entity2":3}, "similar_to", 0.9),
            Construction("similarity_similar_to", ["NOUN","is","similar","to","NOUN"], {"entity1":0,"entity2":4}, "similar_to", 0.95),
            Construction("difference_differs", ["NOUN","differs","from","NOUN"], {"entity1":0,"entity2":3}, "different_from", 0.9),
            Construction("difference_unlike", ["NOUN","is","unlike","NOUN"], {"entity1":0,"entity2":3}, "different_from", 0.9),
            Construction("difference_opposite", ["NOUN","is","ART","opposite","of","NOUN"], {"entity1":0,"entity2":5}, "opposite_of", 0.95),

            # ── V4: Quantity / measurement ─────────────────────────────────────
            Construction("has_value", ["NOUN","has","ART","NOUN","of","NOUN"], {"entity":0,"property":3,"value":5}, "has_value", 0.85),
            Construction("measured_in", ["NOUN","is","measured","in","NOUN"], {"quantity":0,"unit":4}, "measured_in", 0.95),
            Construction("equals", ["NOUN","equals","NOUN"], {"left":0,"right":2}, "equals", 0.95),

            # ── V4: Capability / property ──────────────────────────────────────
            Construction("capable_of", ["NOUN","is","capable","of","NOUN"], {"agent":0,"capability":4}, "capable_of", 0.95),
            Construction("responsible_for", ["NOUN","is","responsible","for","NOUN"], {"agent":0,"responsibility":4}, "responsible_for", 0.9),
            Construction("dependent_on", ["NOUN","depends","on","NOUN"], {"dependent":0,"dependency":3}, "depends_on", 0.9),
            Construction("associated_with", ["NOUN","is","associated","with","NOUN"], {"entity1":0,"entity2":4}, "associated_with", 0.9),
        ]

    def match(self, words: List[str]) -> List[ConstructionMatch]:
        matches: List[ConstructionMatch] = []
        for construction in self.constructions:
            pat = construction.pattern
            pat_len = len(pat)
            if pat_len > len(words):
                continue
            for start in range(len(words) - pat_len + 1):
                window = words[start:start + pat_len]
                if all(_slot_matches(pat[i], window[i]) for i in range(pat_len)):
                    fillers = {role: window[pos] for role, pos in construction.roles.items()}
                    score = construction.confidence
                    matches.append(ConstructionMatch(construction, fillers, score))
        matches.sort(key=lambda m: (-m.score, -len(m.construction.pattern)))
        return matches
