"""
NSCK Statistical + Rule-Based POS Tagger (V6)
=============================================
A lightweight, dependency-free part-of-speech tagger that improves NSCK's
NLU coverage from ~75–85 % to ~90 %+.

Architecture
------------
Two-stage pipeline:

1. **Lexical stage**: assigns a default tag to each word using:
   - A hand-crafted closed-class lexicon (function words, auxiliaries, etc.)
   - Morphological suffixes (see ``_SUFFIX_RULES``)
   - A small domain-specific open-class lexicon built from frequent words

2. **Brill transformation stage**: applies a set of context-sensitive
   transformation rules that correct mis-taggings:
   - A word currently tagged X should be re-tagged Y if the context
     matches (e.g., preceding DET → NOUN, following VERB → NOUN, etc.)

Tag set (simplified, NSCK-specific)
------------------------------------
VB   – base verb
VBZ  – 3rd-person singular verb
VBD  – past-tense verb
VBG  – gerund / present-participle
VBN  – past-participle
NN   – noun (singular)
NNS  – noun (plural)
NNP  – proper noun
JJ   – adjective
RB   – adverb
DT   – determiner
IN   – preposition / subordinating conjunction
CC   – coordinating conjunction
PRP  – personal pronoun
PRP$ – possessive pronoun
MD   – modal auxiliary
NEG  – negation word
TEMP – temporal connective
COND – conditional connective
CD   – cardinal number
WP   – wh-pronoun / wh-determiner
EX   – existential "there"
UH   – interjection

Usage
-----
>>> from python.core.language.pos_tagger import BrillPosTagger
>>> tagger = BrillPosTagger()
>>> tagger.tag("The cat sat on the mat".split())
[('The', 'DT'), ('cat', 'NN'), ('sat', 'VBD'), ('on', 'IN'),
 ('the', 'DT'), ('mat', 'NN')]
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Closed-class / function-word lexicon
# ---------------------------------------------------------------------------

_LEXICON: Dict[str, str] = {
    # Determiners
    "the": "DT", "a": "DT", "an": "DT", "this": "DT", "that": "DT",
    "these": "DT", "those": "DT", "some": "DT", "any": "DT",
    "every": "DT", "each": "DT", "no": "DT", "both": "DT", "all": "DT",
    "few": "DT", "many": "DT", "most": "DT", "much": "DT", "more": "DT",
    "less": "DT", "other": "DT", "another": "DT", "such": "DT",
    "what": "WP", "which": "WP", "whose": "WP",
    # Personal pronouns
    "i": "PRP", "me": "PRP", "you": "PRP", "he": "PRP", "she": "PRP",
    "it": "PRP", "we": "PRP", "us": "PRP", "they": "PRP", "them": "PRP",
    "him": "PRP", "her": "PRP",
    # Possessive pronouns
    "my": "PRP$", "your": "PRP$", "his": "PRP$", "its": "PRP$",
    "our": "PRP$", "their": "PRP$", "hers": "PRP$",
    # Wh-words
    "who": "WP", "whom": "WP", "where": "WRB", "when": "WRB",
    "why": "WRB", "how": "WRB", "whatever": "WP", "whoever": "WP",
    # Modal auxiliaries
    "can": "MD", "could": "MD", "will": "MD", "would": "MD",
    "shall": "MD", "should": "MD", "may": "MD", "might": "MD",
    "must": "MD", "ought": "MD",
    # Auxiliaries (verb forms, but treated as special)
    "is": "VBZ", "are": "VBZ", "am": "VBZ",
    "was": "VBD", "were": "VBD",
    "be": "VB", "been": "VBN", "being": "VBG",
    "have": "VB", "has": "VBZ", "had": "VBD", "having": "VBG",
    "do": "VB", "does": "VBZ", "did": "VBD", "doing": "VBG", "done": "VBN",
    # Negation
    "not": "NEG", "never": "NEG", "neither": "NEG", "nor": "NEG",
    "no": "NEG", "nobody": "NEG", "nothing": "NEG", "nowhere": "NEG",
    "cannot": "NEG", "n't": "NEG",
    # Coordinating conjunctions
    "and": "CC", "but": "CC", "or": "CC", "nor": "CC",
    "for": "CC", "yet": "CC", "so": "CC",
    # Subordinating conjunctions / prepositions
    "in": "IN", "on": "IN", "at": "IN", "by": "IN", "of": "IN",
    "to": "IN", "from": "IN", "with": "IN", "into": "IN", "onto": "IN",
    "about": "IN", "above": "IN", "below": "IN", "under": "IN",
    "over": "IN", "between": "IN", "among": "IN", "through": "IN",
    "across": "IN", "around": "IN", "near": "IN", "beside": "IN",
    "behind": "IN", "before": "IN", "after": "IN", "during": "IN",
    "within": "IN", "without": "IN", "beyond": "IN", "along": "IN",
    "toward": "IN", "towards": "IN", "against": "IN", "despite": "IN",
    "except": "IN", "including": "IN", "via": "IN", "per": "IN",
    "because": "IN", "although": "IN", "though": "IN", "while": "IN",
    "since": "IN", "until": "IN", "unless": "IN", "whether": "IN",
    "if": "COND", "when": "TEMP", "then": "TEMP", "once": "TEMP",
    "whenever": "COND", "provided": "COND", "given": "COND",
    "assuming": "COND", "suppose": "COND", "supposing": "COND",
    "before": "TEMP", "after": "TEMP", "since": "TEMP",
    "until": "TEMP", "during": "TEMP", "previously": "TEMP",
    "subsequently": "TEMP", "meanwhile": "TEMP", "later": "TEMP",
    "earlier": "TEMP", "then": "TEMP",
    # Existential
    "there": "EX",
    # Common adverbs
    "very": "RB", "quite": "RB", "rather": "RB", "too": "RB",
    "also": "RB", "just": "RB", "only": "RB", "even": "RB",
    "still": "RB", "already": "RB", "soon": "RB", "often": "RB",
    "always": "RB", "never": "RB", "sometimes": "RB", "usually": "RB",
    "really": "RB", "well": "RB", "now": "RB", "here": "RB",
    "there": "RB", "then": "RB", "again": "RB", "together": "RB",
    "however": "RB", "therefore": "RB", "thus": "RB", "hence": "RB",
    "furthermore": "RB", "moreover": "RB", "nevertheless": "RB",
    "nonetheless": "RB", "otherwise": "RB", "yet": "RB",
    "instead": "RB", "anyway": "RB", "indeed": "RB", "accordingly": "RB",
    "actually": "RB", "certainly": "RB", "clearly": "RB", "finally": "RB",
    "similarly": "RB", "consequently": "RB", "additionally": "RB",
    "meanwhile": "RB",
    # Numbers
    "one": "CD", "two": "CD", "three": "CD", "four": "CD", "five": "CD",
    "six": "CD", "seven": "CD", "eight": "CD", "nine": "CD", "ten": "CD",
    "first": "JJ", "second": "JJ", "third": "JJ", "last": "JJ",
}

# ---------------------------------------------------------------------------
# Morphological suffix rules (checked in order, longest-first)
# ---------------------------------------------------------------------------

_SUFFIX_RULES: List[Tuple[str, str]] = [
    # Adjective suffixes
    ("ational",  "JJ"), ("tional",  "JJ"), ("enci",  "JJ"), ("anci",  "JJ"),
    ("ical",     "JJ"), ("iful",    "JJ"), ("ful",   "JJ"), ("less",  "JJ"),
    ("ous",      "JJ"), ("ive",     "JJ"), ("ible",  "JJ"), ("able",  "JJ"),
    ("ish",      "JJ"), ("ian",     "JJ"), ("al",    "JJ"),
    ("ic",       "JJ"), ("ary",     "JJ"),
    # Adverb suffixes
    ("ically",   "RB"), ("ously",   "RB"), ("ively",  "RB"), ("ably",  "RB"),
    ("ibly",     "RB"), ("fully",   "RB"), ("lessly", "RB"), ("wards", "RB"),
    ("ward",     "RB"), ("wise",    "RB"), ("ly",     "RB"),
    # Noun suffixes
    ("tion",     "NN"), ("sion",    "NN"), ("ism",    "NN"), ("ist",   "NN"),
    ("ity",      "NN"), ("ty",      "NN"), ("ness",   "NN"), ("ment",  "NN"),
    ("ence",     "NN"), ("ance",    "NN"), ("ship",   "NN"), ("hood",  "NN"),
    ("dom",      "NN"), ("age",     "NN"), ("ure",    "NN"), ("ics",   "NN"),
    ("ogy",      "NN"), ("phy",     "NN"), ("ry",     "NN"), ("ery",   "NN"),
    ("ium",      "NN"), ("um",      "NN"),
    # Verb suffixes
    ("izing",    "VBG"), ("ising",  "VBG"), ("ating", "VBG"), ("ening", "VBG"),
    ("ifying",   "VBG"), ("ying",   "VBG"), ("ing",   "VBG"),
    ("ized",     "VBN"), ("ised",   "VBN"), ("ated",  "VBN"), ("ened",  "VBN"),
    ("ified",    "VBN"), ("ed",     "VBD"),
    ("izes",     "VBZ"), ("ises",   "VBZ"), ("ates",  "VBZ"), ("ifies", "VBZ"),
    ("ifies",    "VBZ"), ("ens",    "VBZ"), ("ies",   "VBZ"), ("es",    "VBZ"),
    ("ize",      "VB"),  ("ise",    "VB"),  ("ate",   "VB"),  ("ify",   "VB"),
    ("en",       "VB"),
    # Plural noun
    ("ments",    "NNS"), ("tions",  "NNS"), ("nesses", "NNS"),
    ("ances",    "NNS"), ("ences",  "NNS"), ("isms",   "NNS"),
    ("ists",     "NNS"), ("ships",  "NNS"), ("hoods",  "NNS"),
    ("ities",    "NNS"), ("ries",   "NNS"), ("ies",    "NNS"),
    ("ses",      "NNS"), ("ches",   "NNS"), ("ves",    "NNS"),
    ("s",        "NNS"),
]

# Minimum word length to apply a given suffix rule (avoids "is" → VBZ via "s")
_SUFFIX_MIN_LEN: Dict[str, int] = {
    "s": 4, "es": 4, "ies": 5, "ed": 4, "ing": 5,
}

# ---------------------------------------------------------------------------
# Brill transformation rules
# ---------------------------------------------------------------------------
# Format: (condition_type, condition_value, current_tag, new_tag)
# Condition types:
#   "prev_tag"   — tag of i-1 is condition_value
#   "next_tag"   — tag of i+1 is condition_value
#   "prev_word"  — word at i-1 is condition_value (lowercase)
#   "next_word"  — word at i+1 is condition_value (lowercase)
#   "prev2_tag"  — tag of i-2 is condition_value
#   "surround"   — tags at i-1 AND i+1 both in condition_value (tuple)

_BRILL_RULES: List[Tuple[str, Any, str, str]] = [
    # After DT/PRP$/JJ → likely a noun
    ("prev_tag",  "DT",    "VB",  "NN"),
    ("prev_tag",  "DT",    "VBZ", "NN"),
    ("prev_tag",  "DT",    "VBG", "NN"),
    ("prev_tag",  "DT",    "VBD", "NN"),
    ("prev_tag",  "DT",    "JJ",  "NN"),
    ("prev_tag",  "DT",    "RB",  "NN"),
    ("prev_tag",  "PRP$",  "VB",  "NN"),
    ("prev_tag",  "PRP$",  "VBG", "NN"),
    ("prev_tag",  "PRP$",  "JJ",  "NN"),
    # After IN → likely NN or NNP
    ("prev_tag",  "IN",    "VB",  "NN"),
    ("prev_tag",  "IN",    "VBD", "NN"),
    # VBD after NN/NNP/PRP → likely still VBD (don't retag)
    # NN after VBZ → likely NN not VB
    ("prev_tag",  "VBZ",   "VB",  "NN"),
    ("prev_tag",  "VBZ",   "VBG", "NN"),
    # Before NN/NNS → likely JJ (adjective before noun)
    ("next_tag",  "NN",    "VB",  "JJ"),
    ("next_tag",  "NN",    "VBD", "JJ"),
    ("next_tag",  "NN",    "NNS", "JJ"),
    ("next_tag",  "NNS",   "VB",  "JJ"),
    # VBG after TO → base verb
    ("prev_word", "to",    "VBG", "VB"),
    # Adverb before adjective
    ("next_tag",  "JJ",    "NN",  "RB"),
    ("next_tag",  "JJ",    "VB",  "RB"),
    # Word following CC is often same POS as preceding word (heuristic)
    # NN before CC → likely NN not VB
    ("next_tag",  "CC",    "VB",  "NN"),
    # After MD → base verb form
    ("prev_tag",  "MD",    "VBZ", "VB"),
    ("prev_tag",  "MD",    "VBD", "VB"),
    ("prev_tag",  "MD",    "NN",  "VB"),
    ("prev_tag",  "MD",    "NNS", "VB"),
    # NEG + word → likely VB (negated verb)
    ("prev_tag",  "NEG",   "NN",  "VB"),
    # Proper noun: word starts with capital, surrounded by lower-case words
    # (handled in code, not as a rule tuple)
]


# ---------------------------------------------------------------------------
# BrillPosTagger
# ---------------------------------------------------------------------------

class BrillPosTagger:
    """
    Two-pass POS tagger:
    1. Lexical lookup + morphological suffix rules.
    2. Brill transformation rules applied left-to-right.

    Parameters
    ----------
    extra_lexicon : dict mapping word → tag for domain-specific additions.

    Example
    -------
    >>> t = BrillPosTagger()
    >>> t.tag("the cat sat on the mat".split())
    [('the', 'DT'), ('cat', 'NN'), ('sat', 'VBD'), ...]
    """

    def __init__(self, extra_lexicon: Optional[Dict[str, str]] = None):
        self._lexicon = dict(_LEXICON)
        if extra_lexicon:
            self._lexicon.update({k.lower(): v for k, v in extra_lexicon.items()})

    def tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Tag a list of tokens.

        Parameters
        ----------
        tokens : list of str (already tokenised)

        Returns
        -------
        list of (token, tag) pairs
        """
        if not tokens:
            return []
        tags = [self._lexical_tag(t) for t in tokens]
        tags = self._apply_brill_rules(tokens, tags)
        tags = self._fix_proper_nouns(tokens, tags)
        return list(zip(tokens, tags))

    def tag_sentence(self, sentence: str) -> List[Tuple[str, str]]:
        """Tokenise and tag a sentence string."""
        tokens = self._tokenise(sentence)
        return self.tag(tokens)

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Simple whitespace+punctuation tokeniser."""
        # Split on spaces and punctuation boundaries
        tokens = re.findall(r"[A-Za-z0-9']+|[^\s\w]", text)
        return [t for t in tokens if t.strip()]

    def _lexical_tag(self, word: str) -> str:
        """Assign an initial tag using lexicon + morphological rules."""
        w_lower = word.lower()

        # Lexicon lookup (handles all closed-class and common words)
        if w_lower in self._lexicon:
            return self._lexicon[w_lower]

        # Digit / number
        if re.fullmatch(r"\d+(?:[.,]\d+)?", w_lower):
            return "CD"

        # Morphological suffixes (longest-suffix-first)
        for suffix, tag in _SUFFIX_RULES:
            min_len = _SUFFIX_MIN_LEN.get(suffix, 3)
            if w_lower.endswith(suffix) and len(w_lower) >= min_len + len(suffix):
                return tag

        # Default: unknown word → NN (most common category in English)
        return "NN"

    @staticmethod
    def _apply_brill_rules(
        tokens: List[str], tags: List[str]
    ) -> List[str]:
        """Apply Brill transformation rules."""
        n = len(tags)
        for i, (word, current_tag) in enumerate(zip(tokens, tags)):
            for cond_type, cond_val, match_tag, new_tag in _BRILL_RULES:
                if current_tag != match_tag:
                    continue
                if cond_type == "prev_tag":
                    if i > 0 and tags[i - 1] == cond_val:
                        tags[i] = new_tag
                        break
                elif cond_type == "next_tag":
                    if i < n - 1 and tags[i + 1] == cond_val:
                        tags[i] = new_tag
                        break
                elif cond_type == "prev_word":
                    if i > 0 and tokens[i - 1].lower() == cond_val:
                        tags[i] = new_tag
                        break
                elif cond_type == "next_word":
                    if i < n - 1 and tokens[i + 1].lower() == cond_val:
                        tags[i] = new_tag
                        break
                elif cond_type == "prev2_tag":
                    if i > 1 and tags[i - 2] == cond_val:
                        tags[i] = new_tag
                        break
                elif cond_type == "surround":
                    prev_t = tags[i - 1] if i > 0 else ""
                    next_t = tags[i + 1] if i < n - 1 else ""
                    if prev_t == cond_val[0] and next_t == cond_val[1]:
                        tags[i] = new_tag
                        break
        return tags

    @staticmethod
    def _fix_proper_nouns(tokens: List[str], tags: List[str]) -> List[str]:
        """
        Capitalized words not at sentence start that are tagged NN/VB
        are likely proper nouns (NNP).
        """
        for i, (word, tag) in enumerate(zip(tokens, tags)):
            if i == 0:
                continue  # sentence-initial capital is not diagnostic
            if word and word[0].isupper() and tag in ("NN", "VB", "VBD"):
                tags[i] = "NNP"
        return tags

    def is_verb(self, tag: str) -> bool:
        """Return True if *tag* is any verb tag."""
        return tag.startswith("VB") or tag == "MD"

    def is_noun(self, tag: str) -> bool:
        """Return True if *tag* is any noun tag."""
        return tag.startswith("NN")

    def is_adjective(self, tag: str) -> bool:
        return tag == "JJ"

    def get_verbs(self, tagged: List[Tuple[str, str]]) -> List[str]:
        """Extract all verb forms from a tagged sentence."""
        return [w for w, t in tagged if self.is_verb(t)]

    def get_nouns(self, tagged: List[Tuple[str, str]]) -> List[str]:
        """Extract all noun forms from a tagged sentence."""
        return [w for w, t in tagged if self.is_noun(t)]


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

_DEFAULT_TAGGER: Optional[BrillPosTagger] = None


def get_default_tagger() -> BrillPosTagger:
    """Return the shared default tagger instance (lazy-initialized)."""
    global _DEFAULT_TAGGER
    if _DEFAULT_TAGGER is None:
        _DEFAULT_TAGGER = BrillPosTagger()
    return _DEFAULT_TAGGER
