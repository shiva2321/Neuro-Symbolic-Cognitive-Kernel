"""Construction Grammar module for NSCK V3."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional

ARTICLES = {"a", "an", "the"}
PREPOSITIONS = {"in", "on", "at", "by", "for", "with", "to", "from", "of", "into", "onto", "about", "above", "below", "under", "over"}
COMMON_VERBS = {
    # Auxiliaries and copulas
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    # Common motion/action verbs
    "run", "runs", "go", "goes", "see", "sees",
    "know", "knows", "think", "thinks", "get", "gets",
    "make", "makes", "come", "comes", "take", "takes",
    "use", "uses", "find", "finds", "give", "gives",
    "tell", "tells", "work", "works", "call", "calls",
    "try", "tries", "ask", "asks", "need", "needs",
    "feel", "feels", "become", "becomes", "leave", "leaves",
    "put", "puts", "mean", "means", "keep", "keeps",
    "let", "lets", "begin", "begins", "show", "shows",
    "hear", "hears", "play", "plays", "move", "moves",
    "pay", "pays", "set", "sets", "change", "changes",
    # Domain-specific relation verbs
    "causes", "cause", "caused",
    "contains", "contain",
    "produce", "produces",
    "require", "requires",
    "enable", "enables",
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
    w = word.lower()
    if w in ARTICLES:
        return "ART"
    if w in PREPOSITIONS:
        return "PREP"
    if w in COMMON_VERBS:
        return "VERB"
    if w.endswith("ed") or w.endswith("ing"):
        return "VERB"
    if w.endswith("ly"):
        return "ADJ"
    if w.endswith("ful") or w.endswith("less") or w.endswith("ous") or w.endswith("ive") or w.endswith("ish"):
        return "ADJ"
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
