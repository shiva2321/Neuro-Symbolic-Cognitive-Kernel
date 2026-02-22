"""Construction Grammar module for NSCK V3."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional

ARTICLES = {"a", "an", "the"}
PREPOSITIONS = {"in", "on", "at", "by", "for", "with", "to", "from", "of", "into", "onto", "about", "above", "below", "under", "over"}
NEGATIONS = {"not", "never", "no", "n't", "cannot", "can't", "won't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"}
COMMON_VERBS = {
    # Auxiliaries and modals
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "done",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    # Motion / location
    "run", "runs", "ran", "running",
    "go", "goes", "went", "gone", "going",
    "come", "comes", "came", "coming",
    "move", "moves", "moved", "moving",
    "leave", "leaves", "left", "leaving",
    "arrive", "arrives", "arrived",
    "reach", "reaches", "reached",
    "enter", "enters", "entered",
    "return", "returns", "returned",
    "travel", "travels", "traveled",
    "walk", "walks", "walked",
    "fly", "flies", "flew", "flown",
    "fall", "falls", "fell", "fallen",
    "rise", "rises", "rose", "risen",
    "grow", "grows", "grew", "grown",
    # Perception / cognition
    "see", "sees", "saw", "seen",
    "hear", "hears", "heard",
    "know", "knows", "knew", "known",
    "think", "thinks", "thought",
    "feel", "feels", "felt",
    "understand", "understands", "understood",
    "learn", "learns", "learned", "learnt",
    "remember", "remembers", "remembered",
    "forget", "forgets", "forgot", "forgotten",
    "believe", "believes", "believed",
    "realize", "realizes", "realized",
    "notice", "notices", "noticed",
    "recognize", "recognizes", "recognized",
    "perceive", "perceives", "perceived",
    "imagine", "imagines", "imagined",
    "observe", "observes", "observed",
    "detect", "detects", "detected",
    # Communication
    "say", "says", "said",
    "tell", "tells", "told",
    "ask", "asks", "asked",
    "call", "calls", "called",
    "speak", "speaks", "spoke", "spoken",
    "talk", "talks", "talked",
    "write", "writes", "wrote", "written",
    "read", "reads",
    "answer", "answers", "answered",
    "explain", "explains", "explained",
    "describe", "describes", "described",
    "report", "reports", "reported",
    "announce", "announces", "announced",
    "suggest", "suggests", "suggested",
    "argue", "argues", "argued",
    "claim", "claims", "claimed",
    "state", "states", "stated",
    "mention", "mentions", "mentioned",
    # Action / manipulation
    "make", "makes", "made",
    "take", "takes", "took", "taken",
    "get", "gets", "got", "gotten",
    "give", "gives", "gave", "given",
    "put", "puts",
    "set", "sets",
    "keep", "keeps", "kept",
    "let", "lets",
    "use", "uses", "used",
    "find", "finds", "found",
    "begin", "begins", "began", "begun",
    "start", "starts", "started",
    "stop", "stops", "stopped",
    "show", "shows", "showed", "shown",
    "play", "plays", "played",
    "pay", "pays", "paid",
    "try", "tries", "tried",
    "need", "needs", "needed",
    "mean", "means", "meant",
    "change", "changes", "changed",
    "work", "works", "worked",
    "live", "lives", "lived",
    "die", "dies", "died",
    "eat", "eats", "ate", "eaten",
    "drink", "drinks", "drank", "drunk",
    "sleep", "sleeps", "slept",
    "open", "opens", "opened",
    "close", "closes", "closed",
    "break", "breaks", "broke", "broken",
    "cut", "cuts",
    "hold", "holds", "held",
    "turn", "turns", "turned",
    "pass", "passes", "passed",
    "bring", "brings", "brought",
    "carry", "carries", "carried",
    "throw", "throws", "threw", "thrown",
    "catch", "catches", "caught",
    "push", "pushes", "pushed",
    "pull", "pulls", "pulled",
    "pull", "pull",
    "lift", "lifts", "lifted",
    "drop", "drops", "dropped",
    "add", "adds", "added",
    "remove", "removes", "removed",
    "join", "joins", "joined",
    "connect", "connects", "connected",
    "separate", "separates", "separated",
    "combine", "combines", "combined",
    "mix", "mixes", "mixed",
    "form", "forms", "formed",
    "appear", "appears", "appeared",
    "happen", "happens", "happened",
    "occur", "occurs", "occurred",
    "exist", "exists", "existed",
    "remain", "remains", "remained",
    # Commerce / production
    "sell", "sells", "sold",
    "buy", "buys", "bought",
    "trade", "trades", "traded",
    "hire", "hires", "hired",
    "build", "builds", "built",
    "create", "creates", "created",
    "send", "sends", "sent",
    "receive", "receives", "received",
    "own", "owns", "owned",
    "pay", "pays", "paid",
    "earn", "earns", "earned",
    "spend", "spends", "spent",
    "cost", "costs",
    "lose", "loses", "lost",
    "win", "wins", "won",
    "gain", "gains", "gained",
    "increase", "increases", "increased",
    "decrease", "decreases", "decreased",
    "reduce", "reduces", "reduced",
    "raise", "raises", "raised",
    "lower", "lowers", "lowered",
    # Relation / causation verbs
    "cause", "causes", "caused",
    "contain", "contains",
    "produce", "produces", "produced",
    "require", "requires", "required",
    "enable", "enables", "enabled",
    "prevent", "prevents", "prevented",
    "support", "supports",
    "include", "includes",
    "follow", "follows",
    "represent", "represents",
    "perform", "performs",
    "discover", "discovers", "discovered",
    "develop", "develops", "developed",
    "define", "defines", "defined",
    "involve", "involves",
    "allow", "allows",
    "affect", "affects", "affected",
    "influence", "influences", "influenced",
    "determine", "determines", "determined",
    "control", "controls", "controlled",
    "manage", "manages", "managed",
    "lead", "leads", "led",
    "result", "results", "resulted",
    "depend", "depends", "depended",
    "apply", "applies", "applied",
    "relate", "relates", "related",
    "belong", "belongs", "belonged",
    "refer", "refers", "referred",
    "indicate", "indicates", "indicated",
    "demonstrate", "demonstrates", "demonstrated",
    "prove", "proves", "proved", "proven",
    "test", "tests", "tested",
    "measure", "measures", "measured",
    "compare", "compares", "compared",
    "differ", "differs", "differed",
    "share", "shares", "shared",
    "interact", "interacts", "interacted",
    "contribute", "contributes", "contributed",
    "replace", "replaces", "replaced",
    "improve", "improves", "improved",
    "enhance", "enhances", "enhanced",
    "expand", "expands", "expanded",
    "extend", "extends", "extended",
    "limit", "limits", "limited",
    "restrict", "restricts", "restricted",
    "protect", "protects", "protected",
    "destroy", "destroys", "destroyed",
    "damage", "damages", "damaged",
    "heal", "heals", "healed",
    "save", "saves", "saved",
    "help", "helps", "helped",
    "hurt", "hurts",
    "attack", "attacks", "attacked",
    "defend", "defends", "defended",
    "love", "loves", "loved",
    "hate", "hates", "hated",
    "like", "likes", "liked",
    "want", "wants", "wanted",
    "hope", "hopes", "hoped",
    "fear", "fears", "feared",
    "accept", "accepts", "accepted",
    "reject", "rejects", "rejected",
    "choose", "chooses", "chose", "chosen",
    "decide", "decides", "decided",
    "plan", "plans", "planned",
    "prepare", "prepares", "prepared",
    "teach", "teaches", "taught",
    "train", "trains", "trained",
    "study", "studies", "studied",
    "research", "researches", "researched",
    "analyze", "analyzes", "analyzed",
    "solve", "solves", "solved",
    "design", "designs", "designed",
    "implement", "implements", "implemented",
    "run", "runs", "ran",
    "execute", "executes", "executed",
    "process", "processes", "processed",
    "store", "stores", "stored",
    "retrieve", "retrieves", "retrieved",
    "access", "accesses", "accessed",
    "share", "shares", "shared",
    "publish", "publishes", "published",
    "update", "updates", "updated",
    "delete", "deletes", "deleted",
    "install", "installs", "installed",
    "activate", "activates", "activated",
    "generate", "generates", "generated",
    "transform", "transforms", "transformed",
    "convert", "converts", "converted",
    "transfer", "transfers", "transferred",
    "distribute", "distributes", "distributed",
    "collect", "collects", "collected",
    "classify", "classifies", "classified",
    "identify", "identifies", "identified",
    "estimate", "estimates", "estimated",
    "predict", "predicts", "predicted",
    "simulate", "simulates", "simulated",
    "model", "models", "modeled",
    "adapt", "adapts", "adapted",
    "evolve", "evolves", "evolved",
    "emerge", "emerges", "emerged",
    "stabilize", "stabilizes", "stabilized",
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
    """Classify a single word into a POS category."""
    w = word.lower().rstrip(".,;!?")
    if w in ARTICLES:
        return "ART"
    if w in PREPOSITIONS:
        return "PREP"
    if w in NEGATIONS:
        return "NEG"
    if w in COMMON_VERBS:
        return "VERB"
    # Morphological verb detection: -ed / -ing suffixes, but guard against
    # nouns ending in -ed/-ing (e.g. "speed", "king", "building" as noun).
    # Heuristic: length >= 5 to avoid false positives on short words.
    if len(w) >= 5 and (w.endswith("ing") or w.endswith("ize") or w.endswith("ise") or w.endswith("ify")):
        return "VERB"
    if len(w) >= 5 and w.endswith("ed") and not w.endswith("speed"):
        return "VERB"
    if w.endswith("ly"):
        return "ADJ"
    if w.endswith("ful") or w.endswith("less") or w.endswith("ous") or w.endswith("ive") or w.endswith("ish"):
        return "ADJ"
    if w.endswith("tion") or w.endswith("sion") or w.endswith("ment") or w.endswith("ness") or w.endswith("ity"):
        return "NOUN"
    return "NOUN"

def _slot_matches(slot: str, word: str) -> bool:
    """Check if a word matches a pattern slot."""
    if slot in ("NOUN", "VERB", "ADJ", "PREP", "ART"):
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
