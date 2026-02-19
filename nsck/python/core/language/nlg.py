"""
NSCK Natural Language Generation (NLG) Module
==============================================
Template-based language generation from cognitive state.

Converts internal NSCK representations (rules, episodic memories,
emotions, causal chains, analogies) into human-readable natural language.

Design
------
* No neural network — uses structured templates with slot-filling.
* Templates are scored by relevance; the best match is chosen.
* Supports: narration, explanation, reflection, prediction, comparison.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import random
import time


# ---------------------------------------------------------------------------
# Template System
# ---------------------------------------------------------------------------

@dataclass
class Template:
    """A fill-in-the-blank template."""
    pattern: str          # e.g. "I learned that {concept_a} causes {concept_b}."
    category: str         # "rule" | "episode" | "emotion" | "causal" | "analogy" | "reflection"
    required_slots: List[str]  # slots that must be present
    priority: float = 1.0


# Pre-built template library
_TEMPLATES: List[Template] = [
    # --- Rules ---
    Template("When {condition} happens, {consequence} usually follows.",
             "rule", ["condition", "consequence"]),
    Template("I discovered a pattern: {condition} leads to {consequence}.",
             "rule", ["condition", "consequence"]),
    Template("Rule learned: if {condition}, then {consequence} (seen {support} times).",
             "rule", ["condition", "consequence", "support"]),

    # --- Episodic Memory ---
    Template("I remember that in {domain}, I tried {action} and got {outcome} (reward {reward}).",
             "episode", ["domain", "action", "outcome", "reward"]),
    Template("A past experience in {domain}: action {action} resulted in {outcome}.",
             "episode", ["domain", "action", "outcome"]),
    Template("Looking back, the most impactful moment was scoring {reward} after {action} in {domain}.",
             "episode", ["domain", "action", "reward"]),

    # --- Emotion ---
    Template("Right now I feel {emotion} (valence {valence}, arousal {arousal}).",
             "emotion", ["emotion", "valence", "arousal"]),
    Template("My emotional state is {emotion} — intensity is {intensity}.",
             "emotion", ["emotion", "intensity"]),
    Template("I'm feeling {emotion}. That input made me shift toward {valence_dir} mood.",
             "emotion", ["emotion", "valence_dir"]),

    # --- Causal Reasoning ---
    Template("{cause} causes {effect}.",
             "causal", ["cause", "effect"]),
    Template("I traced a causal chain: {chain_text}.",
             "causal", ["chain_text"]),
    Template("If {cause} had not happened, {counterfactual} might have occurred instead.",
             "causal", ["cause", "counterfactual"]),

    # --- Analogy / Transfer ---
    Template("I see a connection: {source_concept} in {source_domain} is like {target_concept} in {target_domain}.",
             "analogy", ["source_concept", "target_concept", "source_domain", "target_domain"]),
    Template("Knowledge from {source_domain} transfers to {target_domain} at {similarity} similarity.",
             "analogy", ["source_domain", "target_domain", "similarity"]),

    # --- Self-Reflection ---
    Template("My performance in {domain} is {trend}: confidence {confidence}.",
             "reflection", ["domain", "trend", "confidence"]),
    Template("I've processed {total_inputs} inputs so far and learned {rule_count} rules.",
             "reflection", ["total_inputs", "rule_count"]),
    Template("In {domain}, my accuracy trend is {trend} over the last {window} episodes.",
             "reflection", ["domain", "trend", "window"]),

    # --- Prediction ---
    Template("Given {state}, I predict {prediction} will happen next.",
             "prediction", ["state", "prediction"]),
    Template("Based on my model, the best action is {action} because {reason}.",
             "prediction", ["action", "reason"]),

    # --- Theory of Mind ---
    Template("I believe {agent} thinks {belief}.",
             "tom", ["agent", "belief"]),
    Template("{agent} has a false belief: they think {believed}, but actually {reality}.",
             "tom", ["agent", "believed", "reality"]),

    # --- General / Summary ---
    Template("Summary: {summary}.",
             "summary", ["summary"]),
    Template("Here's what I know so far: {summary}.",
             "summary", ["summary"]),
]


# ---------------------------------------------------------------------------
# NLG Engine
# ---------------------------------------------------------------------------

class NLGEngine:
    """
    Generates natural-language descriptions from NSCK cognitive state.

    Supports two modes:
    1. **Template-based** (fast, deterministic) — slot filling from predefined patterns.
    2. **Markov generative** (flexible, novel) — learned n-gram language model
       that produces novel sentences conditioned on a topic/category.

    Usage::

        nlg = NLGEngine()

        # Template mode (default)
        text = nlg.narrate_rule(condition="food nearby", consequence="move forward")

        # Generative mode
        nlg.learn_corpus(["The agent moved north.", "Reward was high."])
        text = nlg.generate_novel(seed="agent", max_words=15)

        # Free-form from slot dict
        text = nlg.generate(category="episode",
                            slots={"domain": "snake", "action": "UP",
                                   "outcome": "ate food", "reward": 1.0})
    """

    def __init__(self, templates: Optional[List[Template]] = None):
        self.templates = templates or list(_TEMPLATES)
        self._history: List[str] = []
        self._max_history = 200
        # Markov chain bigram model for generative NLG
        self._bigrams: Dict[str, Dict[str, int]] = {}
        self._vocab: set = set()
        self._corpus_size: int = 0

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(self, category: str, slots: Dict[str, Any]) -> str:
        """Pick the best matching template and fill slots.

        Parameters
        ----------
        category : template category filter (e.g. "rule", "emotion")
        slots    : dict of {slot_name: value}

        Returns
        -------
        Filled natural-language string.
        """
        candidates = [t for t in self.templates if t.category == category]
        if not candidates:
            # Fallback
            return self._fallback(category, slots)

        # Score: higher if more required slots are present
        scored: List[Tuple[float, Template]] = []
        for t in candidates:
            present = sum(1 for s in t.required_slots if s in slots)
            total = len(t.required_slots) or 1
            score = (present / total) * t.priority
            if present == total:
                score += 1.0  # bonus for full match
            scored.append((score, t))

        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]

        # Fill
        text = best.pattern
        for slot, val in slots.items():
            placeholder = "{" + slot + "}"
            if placeholder in text:
                text = text.replace(placeholder, str(val))

        # Remove unfilled placeholders
        import re
        text = re.sub(r"\{[a-z_]+\}", "???", text)

        self._record(text)
        return text

    # ------------------------------------------------------------------
    # Convenience narrators
    # ------------------------------------------------------------------
    def narrate_rule(self, condition: str, consequence: str,
                     support: int = 0) -> str:
        slots: Dict[str, Any] = {
            "condition": condition,
            "consequence": consequence,
        }
        if support:
            slots["support"] = support
        return self.generate("rule", slots)

    def narrate_episode(self, domain: str, action: str,
                        outcome: str, reward: float) -> str:
        return self.generate("episode", {
            "domain": domain,
            "action": action,
            "outcome": outcome,
            "reward": f"{reward:+.2f}",
        })

    def narrate_emotion(self, emotion: str, valence: float = 0.0,
                        arousal: float = 0.5, intensity: float = 0.5) -> str:
        valence_dir = "positive" if valence > 0 else "negative" if valence < 0 else "neutral"
        return self.generate("emotion", {
            "emotion": emotion,
            "valence": f"{valence:+.2f}",
            "arousal": f"{arousal:.2f}",
            "intensity": f"{intensity:.2f}",
            "valence_dir": valence_dir,
        })

    def narrate_causal(self, cause: str, effect: str,
                       chain: Optional[List[str]] = None,
                       counterfactual: Optional[str] = None) -> str:
        slots: Dict[str, Any] = {"cause": cause, "effect": effect}
        if chain:
            slots["chain_text"] = " → ".join(chain)
        if counterfactual:
            slots["counterfactual"] = counterfactual
        return self.generate("causal", slots)

    def narrate_analogy(self, source_domain: str, target_domain: str,
                        source_concept: str = "", target_concept: str = "",
                        similarity: float = 0.0) -> str:
        return self.generate("analogy", {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "source_concept": source_concept,
            "target_concept": target_concept,
            "similarity": f"{similarity:.0%}",
        })

    def narrate_reflection(self, domain: str = "", trend: str = "",
                           confidence: float = 0.0, total_inputs: int = 0,
                           rule_count: int = 0, window: int = 10) -> str:
        return self.generate("reflection", {
            "domain": domain,
            "trend": trend,
            "confidence": f"{confidence:.1%}",
            "total_inputs": total_inputs,
            "rule_count": rule_count,
            "window": window,
        })

    def narrate_tom(self, agent: str, belief: str = "",
                    believed: str = "", reality: str = "") -> str:
        return self.generate("tom", {
            "agent": agent,
            "belief": belief,
            "believed": believed,
            "reality": reality,
        })

    def narrate_prediction(self, state: str, prediction: str,
                           action: str = "", reason: str = "") -> str:
        return self.generate("prediction", {
            "state": state,
            "prediction": prediction,
            "action": action,
            "reason": reason,
        })

    # ------------------------------------------------------------------
    # Batch / summary
    # ------------------------------------------------------------------
    def summarise_session(self, stats: Dict[str, Any]) -> str:
        """Generate a session summary from a stats dict."""
        parts = []
        if "total_inputs" in stats:
            parts.append(f"Processed {stats['total_inputs']} inputs")
        if "rule_count" in stats:
            parts.append(f"learned {stats['rule_count']} rules")
        if "domains" in stats:
            parts.append(f"across domains: {', '.join(stats['domains'])}")
        if "best_domain" in stats:
            parts.append(f"best at {stats['best_domain']}")
        summary = "; ".join(parts) if parts else "No data yet"
        return self.generate("summary", {"summary": summary})

    def get_history(self, n: int = 10) -> List[str]:
        """Return the last *n* generated texts."""
        return self._history[-n:]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fallback(self, category: str, slots: Dict[str, Any]) -> str:
        """Fallback when no template matches."""
        pairs = ", ".join(f"{k}={v}" for k, v in slots.items())
        text = f"[{category}] {pairs}"
        self._record(text)
        return text

    def _record(self, text: str):
        self._history.append(text)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def add_template(self, pattern: str, category: str,
                     required_slots: List[str], priority: float = 1.0):
        """Register a custom template at runtime."""
        self.templates.append(Template(
            pattern=pattern,
            category=category,
            required_slots=required_slots,
            priority=priority,
        ))

    # ------------------------------------------------------------------
    # Markov-chain generative NLG
    # ------------------------------------------------------------------
    def learn_corpus(self, sentences: List[str]):
        """Train the bigram language model from a list of sentences.

        Can be called incrementally — new sentences are folded into
        the existing model.
        """
        for sentence in sentences:
            words = sentence.lower().split()
            if len(words) < 2:
                continue
            self._vocab.update(words)
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                if w1 not in self._bigrams:
                    self._bigrams[w1] = {}
                self._bigrams[w1][w2] = self._bigrams[w1].get(w2, 0) + 1
            self._corpus_size += 1

    def generate_novel(self, seed: str = "", max_words: int = 20,
                       category: str = "", slots: Optional[Dict[str, Any]] = None) -> str:
        """Generate a novel sentence using the Markov chain model.

        If the model is untrained or the seed is not in vocabulary,
        falls back to template generation.

        Parameters
        ----------
        seed : starting word (or first word of a category topic)
        max_words : maximum sentence length
        category : optional category for template fallback
        slots : optional slot dict for template fallback
        """
        if not self._bigrams:
            # Markov model not trained — fall back to templates
            if category and slots:
                return self.generate(category, slots)
            return f"[No generative model trained yet. Seed: {seed}]"

        # Find closest seed word in vocabulary
        start = seed.lower().strip().split()[0] if seed else ""
        if start not in self._bigrams:
            # Try partial match
            candidates = [w for w in self._bigrams if start and w.startswith(start[:3])]
            if candidates:
                start = random.choice(candidates)
            else:
                start = random.choice(list(self._bigrams.keys()))

        words = [start]
        current = start
        for _ in range(max_words - 1):
            if current not in self._bigrams:
                break
            nexts = self._bigrams[current]
            total = sum(nexts.values())
            r = random.random() * total
            cumulative = 0.0
            chosen = None
            for w, count in nexts.items():
                cumulative += count
                if cumulative >= r:
                    chosen = w
                    break
            if chosen is None:
                break
            words.append(chosen)
            current = chosen
            # Stop at sentence boundary
            if chosen.endswith(".") or chosen.endswith("!") or chosen.endswith("?"):
                break

        text = " ".join(words)
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        self._record(text)
        return text

    @property
    def is_generative(self) -> bool:
        """Whether the engine has a trained generative model."""
        return len(self._bigrams) > 0
