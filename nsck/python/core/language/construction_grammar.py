"""Construction Grammar module for NSCK V3."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional

ARTICLES = {"a", "an", "the"}
PREPOSITIONS = {"in", "on", "at", "by", "for", "with", "to", "from", "of", "into", "onto", "about", "above", "below", "under", "over", "through", "between", "among", "within", "without", "during", "before", "after", "against", "across", "along", "behind", "beside", "beyond", "down", "up", "around", "near"}
# Negation particles — never classified as NOUN, VERB, ADJ, etc.
_NEGATORS = frozenset({"not", "no", "never", "neither", "nor", "cannot", "n't"})
# Nouns that end in -ing (gerund-nouns) and must NOT be classified as verbs.
# Extended from common English lexicon — add more as needed.
_ING_NOUNS = frozenset({
    "flooding", "building", "morning", "evening", "warning", "opening", "ending",
    "setting", "reading", "writing", "meeting", "feeling", "meaning", "heading",
    "training", "learning", "leading", "housing", "clothing", "funding", "parking",
    "farming", "mining", "fishing", "hunting", "cooking", "painting", "drawing",
    "marketing", "planning", "testing", "coding", "reasoning", "processing",
    "computing", "engineering", "printing", "mapping", "logging", "boarding",
    "binding", "grounding", "landing", "standing", "understanding", "beginning",
    "spring", "string", "ring", "king", "sing", "wing", "thing", "swing",
    "timing", "pricing", "networking", "sorting", "ranking", "listing",
    "thinking", "feeling", "being", "meaning", "happening", "belonging",
    "pricing", "staffing", "shipping", "billing", "accounting", "reporting",
    "wording", "funding", "backing", "following", "offering", "dealing",
    "voting", "hearing", "teaching", "coaching", "typing", "filing",
})
# Past-participle / -ed words used as adjectives or nouns (not verbs in isolation)
_ED_ADJECTIVES = frozenset({
    "advanced", "aged", "armed", "beloved", "blessed", "bored", "broken",
    "burned", "colored", "complicated", "concerned", "confused", "connected",
    "controlled", "crooked", "crowded", "curved", "damaged", "dedicated",
    "defined", "delighted", "depressed", "developed", "directed", "divided",
    "educated", "employed", "engaged", "enhanced", "excited", "experienced",
    "extended", "famed", "fixed", "focused", "forced", "formed", "frightened",
    "frustrated", "gifted", "guided", "highly-skilled", "informed", "injured",
    "integrated", "intended", "interested", "involved", "isolated", "known",
    "learned", "limited", "linked", "locked", "loved", "named", "needed",
    "noted", "observed", "organized", "owned", "pleased", "prepared",
    "published", "qualified", "raised", "recorded", "reduced", "registered",
    "related", "relaxed", "required", "restricted", "skilled", "structured",
    "suited", "supposed", "tired", "trained", "trusted", "unexpected",
    "unified", "united", "varied", "worried", "wired", "dedicated",
})
COMMON_VERBS = {
    # ── Auxiliaries and copulas ──────────────────────────────────────────────
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might",
    "can", "could", "must", "ought",
    # ── Motion / existence ───────────────────────────────────────────────────
    "run", "runs", "ran", "go", "goes", "went", "gone",
    "come", "comes", "came", "move", "moves", "moved",
    "travel", "travels", "traveled", "arrive", "arrives",
    "leave", "leaves", "left", "reach", "reaches", "reached",
    "return", "returns", "returned", "approach", "approaches",
    "enter", "enters", "entered", "exit", "exits", "exited",
    "rise", "rises", "rose", "risen", "fall", "falls", "fell",
    "fly", "flies", "flew", "flown", "swim", "swims", "swam",
    "walk", "walks", "walked", "climb", "climbs", "climbed",
    "stand", "stands", "stood", "sit", "sits", "sat",
    "lie", "lies", "lay", "lain", "stay", "stays", "stayed",
    # ── Perception ───────────────────────────────────────────────────────────
    "see", "sees", "saw", "seen",
    "hear", "hears", "heard",
    "feel", "feels", "felt",
    "smell", "smells", "smelled", "taste", "tastes", "tasted",
    "watch", "watches", "watched", "observe", "observes", "observed",
    "notice", "notices", "noticed", "detect", "detects", "detected",
    "sense", "senses", "sensed", "perceive", "perceives", "perceived",
    # ── Cognition ────────────────────────────────────────────────────────────
    "know", "knows", "knew", "known",
    "think", "thinks", "thought",
    "believe", "believes", "believed",
    "understand", "understands", "understood",
    "realize", "realizes", "realized",
    "learn", "learns", "learned",
    "remember", "remembers", "remembered",
    "forget", "forgets", "forgot",
    "decide", "decides", "decided",
    "consider", "considers", "considered",
    "assume", "assumes", "assumed",
    "expect", "expects", "expected",
    "imagine", "imagines", "imagined",
    "guess", "guesses", "guessed",
    "suppose", "supposes", "supposed",
    "wonder", "wonders", "wondered",
    "recognize", "recognizes", "recognized",
    "reason", "reasons", "reasoned",
    "calculate", "calculates", "calculated",
    "estimate", "estimates", "estimated",
    "predict", "predicts", "predicted",
    "infer", "infers", "inferred",
    "conclude", "concludes", "concluded",
    "analyse", "analyses", "analyzed", "analyze",
    # ── Communication ────────────────────────────────────────────────────────
    "say", "says", "said",
    "tell", "tells", "told",
    "speak", "speaks", "spoke", "spoken",
    "talk", "talks", "talked",
    "ask", "asks", "asked",
    "answer", "answers", "answered",
    "explain", "explains", "explained",
    "describe", "describes", "described",
    "call", "calls", "called",
    "write", "writes", "wrote", "written",
    "read", "reads",
    "inform", "informs", "informed",
    "announce", "announces", "announced",
    "state", "states", "stated",
    "claim", "claims", "claimed",
    "argue", "argues", "argued",
    "suggest", "suggests", "suggested",
    "propose", "proposes", "proposed",
    "report", "reports", "reported",
    "mention", "mentions", "mentioned",
    "confirm", "confirms", "confirmed",
    "deny", "denies", "denied",
    "promise", "promises", "promised",
    "warn", "warns", "warned",
    "request", "requests", "requested",
    "reply", "replies", "replied",
    # ── Emotion / desire ─────────────────────────────────────────────────────
    "want", "wants", "wanted",
    "need", "needs", "needed",
    "like", "likes", "liked",
    "love", "loves", "loved",
    "hate", "hates", "hated",
    "enjoy", "enjoys", "enjoyed",
    "prefer", "prefers", "preferred",
    "fear", "fears", "feared",
    "wish", "wishes", "wished",
    "hope", "hopes", "hoped",
    "care", "cares", "cared",
    "try", "tries", "tried",
    "attempt", "attempts", "attempted",
    "plan", "plans", "planned",
    "intend", "intends", "intended",
    "desire", "desires", "desired",
    # ── Manipulation / action ────────────────────────────────────────────────
    "get", "gets", "got", "gotten",
    "make", "makes", "made",
    "take", "takes", "took", "taken",
    "give", "gives", "gave", "given",
    "use", "uses", "used",
    "find", "finds", "found",
    "put", "puts",
    "set", "sets",
    "keep", "keeps", "kept",
    "let", "lets",
    "mean", "means", "meant",
    "show", "shows", "showed", "shown",
    "play", "plays", "played",
    "pay", "pays", "paid",
    "change", "changes", "changed",
    "hold", "holds", "held",
    "bring", "brings", "brought",
    "carry", "carries", "carried",
    "turn", "turns", "turned",
    "open", "opens", "opened",
    "close", "closes", "closed",
    "start", "starts", "started",
    "stop", "stops", "stopped",
    "finish", "finishes", "finished",
    "begin", "begins", "began", "begun",
    "end", "ends", "ended",
    "add", "adds", "added",
    "remove", "removes", "removed",
    "place", "places", "placed",
    "pick", "picks", "picked",
    "drop", "drops", "dropped",
    "push", "pushes", "pushed",
    "pull", "pulls", "pulled",
    "lift", "lifts", "lifted",
    "cut", "cuts",
    "break", "breaks", "broke", "broken",
    "fix", "fixes", "fixed",
    "form", "forms", "formed",
    "connect", "connects", "connected",
    "separate", "separates", "separated",
    "combine", "combines", "combined",
    "mix", "mixes", "mixed",
    "split", "splits", "split",
    "join", "joins", "joined",
    "link", "links", "linked",
    "attach", "attaches", "attached",
    "apply", "applies", "applied",
    "compare", "compares", "compared",
    "measure", "measures", "measured",
    "test", "tests", "tested",
    "check", "checks", "checked",
    "control", "controls", "controlled",
    "manage", "manages", "managed",
    "lead", "leads", "led",
    "direct", "directs", "directed",
    "drive", "drives", "drove", "driven",
    "handle", "handles", "handled",
    "operate", "operates", "operated",
    "run", "runs", "ran",
    "launch", "launches", "launched",
    "execute", "executes", "executed",
    "process", "processes", "processed",
    "produce", "produces", "produced",
    "generate", "generates", "generated",
    "create", "creates", "created",
    "design", "designs", "designed",
    "implement", "implements", "implemented",
    "install", "installs", "installed",
    "deploy", "deploys", "deployed",
    "access", "accesses", "accessed",
    "store", "stores", "stored",
    "load", "loads", "loaded",
    "save", "saves", "saved",
    "send", "sends", "sent",
    "receive", "receives", "received",
    "transfer", "transfers", "transferred",
    "convert", "converts", "converted",
    "transform", "transforms", "transformed",
    # ── Commerce / creation ──────────────────────────────────────────────────
    "sell", "sells", "sold",
    "buy", "buys", "bought",
    "trade", "trades", "traded",
    "hire", "hires", "hired",
    "build", "builds", "built",
    "construct", "constructs", "constructed",
    "develop", "develops", "developed",
    "grow", "grows", "grew", "grown",
    "expand", "expands", "expanded",
    "increase", "increases", "increased",
    "decrease", "decreases", "decreased",
    "reduce", "reduces", "reduced",
    "improve", "improves", "improved",
    "optimize", "optimizes", "optimized",
    "achieve", "achieves", "achieved",
    "accomplish", "accomplishes", "accomplished",
    "complete", "completes", "completed",
    "earn", "earns", "earned",
    "gain", "gains", "gained",
    "lose", "loses", "lost",
    "win", "wins", "won",
    "succeed", "succeeds", "succeeded",
    "fail", "fails", "failed",
    "support", "supports", "supported",
    "help", "helps", "helped",
    "assist", "assists", "assisted",
    "enable", "enables", "enabled",
    "allow", "allows", "allowed",
    "permit", "permits", "permitted",
    "prevent", "prevents", "prevented",
    "block", "blocks", "blocked",
    "protect", "protects", "protected",
    "serve", "serves", "served",
    "provide", "provides", "provided",
    "offer", "offers", "offered",
    "supply", "supplies", "supplied",
    "deliver", "delivers", "delivered",
    "distribute", "distributes", "distributed",
    # ── Domain-specific relation verbs ───────────────────────────────────────
    "causes", "cause", "caused",
    "contains", "contain", "contained",
    "requires", "require", "required",
    "includes", "include", "included",
    "follows", "follow", "followed",
    "represents", "represent", "represented",
    "performs", "perform", "performed",
    "discovers", "discover", "discovered",
    "defines", "define", "defined",
    "involves", "involve", "involved",
    "exists", "exist", "existed",
    "occurs", "occur", "occurred",
    "happens", "happen", "happened",
    "affects", "affect", "affected",
    "impacts", "impact", "impacted",
    "determines", "determine", "determined",
    "depends", "depend", "depended",
    "varies", "vary", "varied",
    "differs", "differ", "differed",
    "relates", "relate", "related",
    "interacts", "interact", "interacted",
    "works", "work", "worked",
    "functions", "function", "functioned",
    "acts", "act", "acted",
    "behaves", "behave", "behaved",
    "responds", "respond", "responded",
    "reacts", "react", "reacted",
    "absorbs", "absorb", "absorbed",
    "emits", "emit", "emitted",
    "reflects", "reflect", "reflected",
    "conducts", "conduct", "conducted",
    "insulates", "insulate", "insulated",
    "flows", "flow", "flowed",
    "moves", "move", "moved",
    "evolves", "evolve", "evolved",
    "adapts", "adapt", "adapted",
    "learns", "learns",
    "trains", "train", "trained",
    "classifies", "classify", "classified",
    "predicts", "predict", "predicted",
    "solves", "solve", "solved",
    "computes", "compute", "computed",
    "simulates", "simulate", "simulated",
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

    Priority order:
      1. Fixed closed-class sets (articles, prepositions)
      2. Known verb forms (COMMON_VERBS lookup — fastest, most reliable)
      3. Suffix heuristics (order matters: nouns before verb-suffix catches)
         - Known -ing nouns (gerund-nouns) → NOUN
         - Known -ed adjectives → ADJ
         - -ly adverbs → ADJ (treated as ADJ slot)
         - adjectival suffixes → ADJ
         - -tion/-ness/-ity/-ment → NOUN
         - -ise/-ize/-ate/-ify (derivational verb suffixes) → VERB
         - -ed/-ing NOT in above lists → VERB
      4. Default → NOUN
    """
    w = word.lower()
    if w in ARTICLES:
        return "ART"
    if w in PREPOSITIONS:
        return "PREP"
    if w in _NEGATORS:
        return "NEG"
    if w in COMMON_VERBS:
        return "VERB"
    # Gerund-nouns and nominalised -ing words must not be classified as verbs
    if w in _ING_NOUNS:
        return "NOUN"
    # Known -ed adjectives
    if w in _ED_ADJECTIVES:
        return "ADJ"
    # Nominal suffixes → NOUN (check before -ing/-ed verb rule)
    if w.endswith(("tion", "sion", "ness", "ment", "ity", "ism", "ology",
                   "graphy", "ance", "ence", "hood", "ship", "dom")):
        return "NOUN"
    # Adjectival suffixes
    if w.endswith("ly"):
        return "ADJ"
    if w.endswith(("ful", "less", "ous", "ive", "ish", "able", "ible")):
        return "ADJ"
    # Unambiguous adjective-forming suffixes:
    # -ial (essential, official, financial), -ual (actual, virtual, visual),
    # -ical (physical, typical, logical)
    if w.endswith(("ial", "ual", "ical")):
        return "ADJ"
    # Derivational verb suffixes (verb-forming) — note: 'en' excluded as too
    # ambiguous (oxygen, chicken, open, broken all end in 'en')
    if w.endswith(("ise", "ize", "ate", "ify")):
        return "VERB"
    # -ed / -ing not caught above → verb-form heuristic
    if w.endswith("ing") or w.endswith("ed"):
        return "VERB"
    return "NOUN"

def _slot_matches(slot: str, word: str) -> bool:
    """Check if a word matches a pattern slot."""
    if slot in ("NOUN", "VERB", "ADJ", "PREP", "ART", "NEG"):
        return _classify_word(word) == slot
    return word.lower() == slot.lower()

class ConstructionMatcher:
    def __init__(self):
        self.constructions: List[Construction] = self._load_core_constructions()

    def _load_core_constructions(self) -> List[Construction]:
        return [
            Construction("SVO_active", ["NOUN","VERB","NOUN"], {"subject":0,"verb":1,"object":2}, "relates_to", 0.8),
            Construction("copular_is", ["NOUN","is","NOUN"], {"subject":0,"attribute":2}, "is_a", 0.95),
            Construction("copular_is_a", ["NOUN","is","ART","NOUN"], {"subject":0,"attribute":3}, "is_a", 0.95),
            Construction("copular_is_adj", ["NOUN","is","ADJ"], {"subject":0,"quality":2}, "has_property", 0.9),
            Construction("copular_was_adj", ["NOUN","was","ADJ"], {"subject":0,"quality":2}, "has_property", 0.85),
            Construction("copular_are_adj", ["NOUN","are","ADJ"], {"subject":0,"quality":2}, "has_property", 0.9),
            Construction("possessive_has", ["NOUN","has","NOUN"], {"owner":0,"owned":2}, "has_property", 0.95),
            Construction("possessive_have", ["NOUN","have","NOUN"], {"owner":0,"owned":2}, "has_property", 0.9),
            Construction("causative", ["NOUN","causes","NOUN"], {"cause":0,"effect":2}, "causes", 0.95),
            Construction("causative_leads", ["NOUN","leads","to","NOUN"], {"cause":0,"effect":3}, "leads_to", 0.9),
            Construction("causative_results", ["NOUN","results","in","NOUN"], {"cause":0,"effect":3}, "results_in", 0.9),
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
            # ── New V4 constructions ──────────────────────────────────────────
            # Negation
            Construction("negation_cannot", ["NOUN","cannot","VERB","NOUN"], {"subject":0,"verb":2,"object":3}, "cannot_relate_to", 0.9),
            Construction("negation_cannot_vi", ["NOUN","cannot","VERB"], {"subject":0,"action":2}, "cannot_do", 0.9),
            Construction("negation_does_not", ["NOUN","does","not","VERB","NOUN"], {"subject":0,"verb":3,"object":4}, "not_relates_to", 0.9),
            Construction("negation_does_not_vi", ["NOUN","does","not","VERB"], {"subject":0,"action":3}, "cannot_do", 0.9),
            Construction("negation_is_not", ["NOUN","is","not","NOUN"], {"subject":0,"attribute":3}, "is_not", 0.95),
            Construction("negation_is_not_a", ["NOUN","is","not","ART","NOUN"], {"subject":0,"attribute":4}, "is_not", 0.95),
            Construction("negation_is_not_adj", ["NOUN","is","not","ADJ"], {"subject":0,"quality":3}, "lacks_property", 0.9),
            # Conditional / implication
            Construction("conditional_if", ["if","NOUN","VERB","NOUN","then","NOUN","VERB"], {"condition_subj":1,"condition_verb":2,"condition_obj":3,"consequence_subj":5,"consequence_verb":6}, "implies", 0.85),
            Construction("conditional_when", ["when","NOUN","VERB","NOUN","VERB"], {"trigger":2,"action":4}, "triggers", 0.8),
            # Similarity / analogy
            Construction("similarity_like", ["NOUN","is","like","NOUN"], {"entity":0,"reference":3}, "similar_to", 0.9),
            Construction("similarity_similar", ["NOUN","is","ADJ","to","NOUN"], {"entity":0,"reference":4}, "similar_to", 0.85),
            # Temporal
            Construction("temporal_before", ["NOUN","VERB","before","NOUN"], {"event":0,"verb":1,"prior":3}, "precedes", 0.85),
            Construction("temporal_after", ["NOUN","VERB","after","NOUN"], {"event":0,"verb":1,"subsequent":3}, "follows", 0.85),
            # Capability / property
            Construction("can_do", ["NOUN","can","VERB","NOUN"], {"subject":0,"ability":2,"object":3}, "can_do", 0.9),
            Construction("capable_of", ["NOUN","is","capable","of","NOUN"], {"subject":0,"capability":4}, "capable_of", 0.9),
            # Quantification
            Construction("all_are", ["all","NOUN","are","NOUN"], {"universal":1,"category":3}, "is_a", 0.9),
            Construction("most_are", ["most","NOUN","are","ADJ"], {"subject":1,"quality":3}, "typically_has_property", 0.8),
            # Transformation
            Construction("becomes", ["NOUN","becomes","NOUN"], {"before":0,"after":2}, "becomes", 0.9),
            Construction("transforms_into", ["NOUN","VERB","into","NOUN"], {"before":0,"verb":1,"after":3}, "transforms_into", 0.85),
            # Goal / objective
            Construction("tries_to", ["NOUN","tries","to","VERB","NOUN"], {"agent":0,"goal_verb":3,"goal_obj":4}, "has_goal", 0.85),
            Construction("tries_to_vi", ["NOUN","tries","to","VERB"], {"agent":0,"action":3}, "attempts_to", 0.85),
            Construction("wants_to", ["NOUN","wants","to","VERB","NOUN"], {"agent":0,"goal_verb":3,"goal_obj":4}, "wants_to_do", 0.85),
            Construction("wants_to_vi", ["NOUN","wants","to","VERB"], {"agent":0,"action":3}, "wants_to_do", 0.85),
            Construction("needs_to", ["NOUN","needs","to","VERB","NOUN"], {"agent":0,"goal_verb":3,"goal_obj":4}, "needs_to_do", 0.85),
            Construction("needs_to_vi", ["NOUN","needs","to","VERB"], {"agent":0,"action":3}, "needs_to_do", 0.85),
            # Measure / comparison
            Construction("equal_to", ["NOUN","is","equal","to","NOUN"], {"lhs":0,"rhs":4}, "equals", 0.95),
            Construction("greater_than", ["NOUN","is","greater","than","NOUN"], {"larger":0,"smaller":4}, "greater_than", 0.95),
            Construction("less_than", ["NOUN","is","less","than","NOUN"], {"smaller":0,"larger":4}, "less_than", 0.95),
            # Composition
            Construction("composed_of", ["NOUN","is","composed","of","NOUN"], {"whole":0,"part":4}, "composed_of", 0.95),
            Construction("part_of", ["NOUN","is","ART","NOUN","of","NOUN"], {"part":0,"whole":5}, "part_of", 0.9),
            Construction("belongs_to", ["NOUN","belongs","to","NOUN"], {"member":0,"group":3}, "part_of", 0.9),
            # Dependency
            Construction("depends_on", ["NOUN","depends","on","NOUN"], {"dependent":0,"dependency":3}, "depends_on", 0.9),
            Construction("relies_on", ["NOUN","relies","on","NOUN"], {"dependent":0,"dependency":3}, "depends_on", 0.9),
            # Origin
            Construction("comes_from", ["NOUN","comes","from","NOUN"], {"effect":0,"origin":3}, "originates_from", 0.9),
            Construction("derived_from", ["NOUN","is","derived","from","NOUN"], {"derived":0,"source":4}, "derived_from", 0.95),
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
