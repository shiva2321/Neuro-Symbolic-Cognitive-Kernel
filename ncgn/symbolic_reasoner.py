"""
Symbolic Reasoner for System 2 (Logical/Deliberate Processing)
Implements rule-based reasoning with knowledge graphs.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """Represents a fact in the knowledge base"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        return (self.subject == other.subject and
                self.predicate == other.predicate and
                self.object == other.object)


@dataclass
class Rule:
    """Represents a logical rule"""
    premises: List[Tuple[str, str, str]]  # [(subj, pred, obj), ...]
    conclusion: Tuple[str, str, str]
    confidence: float = 1.0
    name: str = ""

    def can_apply(self, facts: Set[Fact]) -> bool:
        """Check if all premises are satisfied"""
        for premise in self.premises:
            found = False
            for fact in facts:
                if self._matches(premise, fact):
                    found = True
                    break
            if not found:
                return False
        return True

    def _matches(self, pattern: Tuple, fact: Fact) -> bool:
        """Check if a pattern matches a fact (supports variables)"""
        subj, pred, obj = pattern

        # Variables start with '?'
        subj_match = subj.startswith('?') or subj == fact.subject
        pred_match = pred.startswith('?') or pred == fact.predicate
        obj_match = obj.startswith('?') or obj == fact.object

        return subj_match and pred_match and obj_match


class KnowledgeBase:
    """
    Knowledge base for storing facts and rules.
    """

    def __init__(self):
        """Initialize empty knowledge base"""
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []

        # Index for efficient lookup
        self.fact_index: Dict[str, Set[Fact]] = defaultdict(set)

    def add_fact(self, subject: str, predicate: str, obj: str,
                 confidence: float = 1.0):
        """Add a fact to the knowledge base"""
        fact = Fact(subject, predicate, obj, confidence)

        if fact not in self.facts:
            self.facts.add(fact)
            # Index by subject and object for faster lookup
            self.fact_index[subject].add(fact)
            self.fact_index[obj].add(fact)

    def add_rule(self, rule: Rule):
        """Add a rule to the knowledge base"""
        self.rules.append(rule)

    def query(self, subject: Optional[str] = None,
              predicate: Optional[str] = None,
              obj: Optional[str] = None) -> List[Fact]:
        """
        Query facts from the knowledge base.

        Args:
            subject: Subject to match (None for wildcard)
            predicate: Predicate to match
            obj: Object to match

        Returns:
            List of matching facts
        """
        results = []

        # Start with all facts or indexed subset
        candidates = self.facts
        if subject:
            candidates = self.fact_index.get(subject, set())
        elif obj:
            candidates = self.fact_index.get(obj, set())

        for fact in candidates:
            match = True
            if subject and fact.subject != subject:
                match = False
            if predicate and fact.predicate != predicate:
                match = False
            if obj and fact.object != obj:
                match = False

            if match:
                results.append(fact)

        return results

    def get_related_entities(self, entity: str, max_hops: int = 2) -> Set[str]:
        """
        Get all entities related to a given entity within max_hops.

        Args:
            entity: Starting entity
            max_hops: Maximum number of hops

        Returns:
            Set of related entities
        """
        related = {entity}
        frontier = {entity}

        for _ in range(max_hops):
            new_frontier = set()

            for e in frontier:
                # Find all facts involving this entity
                facts = self.fact_index.get(e, set())

                for fact in facts:
                    if fact.subject == e:
                        new_frontier.add(fact.object)
                    if fact.object == e:
                        new_frontier.add(fact.subject)

            related.update(new_frontier)
            frontier = new_frontier

            if not frontier:
                break

        return related


class ForwardChaining:
    """
    Forward chaining inference engine.
    Derives new facts from existing facts and rules.
    """

    def __init__(self, kb: KnowledgeBase, max_iterations: int = 10):
        """
        Initialize forward chaining.

        Args:
            kb: Knowledge base
            max_iterations: Maximum inference iterations
        """
        self.kb = kb
        self.max_iterations = max_iterations

    def infer(self) -> Set[Fact]:
        """
        Perform forward chaining inference.

        Returns:
            Set of newly inferred facts
        """
        new_facts = set()

        for iteration in range(self.max_iterations):
            derived_this_round = set()

            # Try to apply each rule
            for rule in self.kb.rules:
                if rule.can_apply(self.kb.facts):
                    # Create binding for variables
                    bindings = self._get_bindings(rule)

                    for binding in bindings:
                        # Apply bindings to conclusion
                        conclusion = self._apply_bindings(rule.conclusion, binding)

                        # Create new fact
                        new_fact = Fact(
                            conclusion[0],
                            conclusion[1],
                            conclusion[2],
                            confidence=rule.confidence
                        )

                        if new_fact not in self.kb.facts:
                            derived_this_round.add(new_fact)

            # Add derived facts to KB
            for fact in derived_this_round:
                self.kb.add_fact(fact.subject, fact.predicate,
                               fact.object, fact.confidence)
                new_facts.add(fact)

            # Stop if no new facts derived
            if not derived_this_round:
                break

            logger.debug(f"Iteration {iteration + 1}: derived {len(derived_this_round)} facts")

        return new_facts

    def _get_bindings(self, rule: Rule) -> List[Dict[str, str]]:
        """Get all possible variable bindings for a rule"""
        # Simplified: only support single variable bindings
        bindings = [{}]

        for premise in rule.premises:
            new_bindings = []

            for binding in bindings:
                # Find facts matching this premise
                matching_facts = self._find_matching_facts(premise, binding)

                for fact in matching_facts:
                    # Create new binding
                    new_binding = binding.copy()

                    # Bind variables
                    if premise[0].startswith('?'):
                        new_binding[premise[0]] = fact.subject
                    if premise[2].startswith('?'):
                        new_binding[premise[2]] = fact.object

                    new_bindings.append(new_binding)

            bindings = new_bindings

        return bindings

    def _find_matching_facts(self, pattern: Tuple, binding: Dict) -> List[Fact]:
        """Find facts matching a pattern with current bindings"""
        subj = binding.get(pattern[0], pattern[0]) if pattern[0].startswith('?') else pattern[0]
        pred = pattern[1]
        obj = binding.get(pattern[2], pattern[2]) if pattern[2].startswith('?') else pattern[2]

        # Query KB
        return self.kb.query(
            subject=None if subj.startswith('?') else subj,
            predicate=pred,
            obj=None if obj.startswith('?') else obj
        )

    def _apply_bindings(self, pattern: Tuple, binding: Dict) -> Tuple:
        """Apply variable bindings to a pattern"""
        return tuple(
            binding.get(item, item) if isinstance(item, str) and item.startswith('?') else item
            for item in pattern
        )


class SymbolicReasoner(nn.Module):
    """
    Symbolic reasoning system for System 2 processing.
    Maintains a knowledge graph and performs logical inference.
    """

    def __init__(self, reasoning_depth: int = 5):
        """
        Initialize symbolic reasoner.

        Args:
            reasoning_depth: Maximum inference depth
        """
        super().__init__()

        self.reasoning_depth = reasoning_depth
        self.kb = KnowledgeBase()
        self.inference_engine = ForwardChaining(self.kb, max_iterations=reasoning_depth)

        # Initialize with common sense rules
        self._initialize_common_rules()

    def _initialize_common_rules(self):
        """Initialize with basic commonsense rules"""
        # Transitivity of "is_a"
        self.kb.add_rule(Rule(
            premises=[
                ('?x', 'is_a', '?y'),
                ('?y', 'is_a', '?z')
            ],
            conclusion=('?x', 'is_a', '?z'),
            confidence=0.9,
            name="transitivity_is_a"
        ))

        # Symmetry of "similar_to"
        self.kb.add_rule(Rule(
            premises=[('?x', 'similar_to', '?y')],
            conclusion=('?y', 'similar_to', '?x'),
            confidence=1.0,
            name="symmetry_similar_to"
        ))

        logger.info("Initialized symbolic reasoner with common rules")

    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 1.0):
        """Add a fact to the knowledge base"""
        self.kb.add_fact(subject, predicate, obj, confidence)

    def add_rule(self, rule: Rule):
        """Add a reasoning rule"""
        self.kb.add_rule(rule)

    def reason(self, query: Optional[Tuple[str, str, str]] = None) -> List[Fact]:
        """
        Perform reasoning to derive new facts.

        Args:
            query: Optional query tuple (subject, predicate, object)

        Returns:
            List of relevant facts
        """
        # Perform forward chaining
        new_facts = self.inference_engine.infer()

        logger.info(f"Derived {len(new_facts)} new facts through reasoning")

        # If query provided, return matching facts
        if query:
            return self.kb.query(query[0], query[1], query[2])
        else:
            return list(new_facts)

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for integration with neural systems.

        Args:
            context: Context dictionary with query information

        Returns:
            Reasoning results
        """
        # Extract query from context
        query_entity = context.get('entity', None)

        if query_entity:
            # Get related facts
            related_entities = self.kb.get_related_entities(query_entity, max_hops=2)
            facts = []

            for entity in related_entities:
                facts.extend(self.kb.query(subject=entity))
                facts.extend(self.kb.query(obj=entity))

            return {
                'facts': facts,
                'related_entities': related_entities,
                'confidence': 1.0 if facts else 0.0
            }
        else:
            # General reasoning
            new_facts = self.reason()

            return {
                'facts': new_facts,
                'confidence': 0.5
            }

    def get_explanation(self, fact: Fact) -> List[str]:
        """
        Generate an explanation for how a fact was derived.

        Args:
            fact: Fact to explain

        Returns:
            List of explanation steps
        """
        explanation = []

        # Check if fact is directly in KB
        if fact in self.kb.facts:
            explanation.append(f"Direct fact: {fact.subject} {fact.predicate} {fact.object}")
        else:
            explanation.append(f"Cannot explain: fact not in knowledge base")

        return explanation

    def save(self, path: str):
        """Save knowledge base to file"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'facts': self.kb.facts,
                'rules': self.kb.rules
            }, f)
        logger.info(f"Saved knowledge base to {path}")

    def load(self, path: str):
        """Load knowledge base from file"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.kb.facts = data['facts']
        self.kb.rules = data['rules']

        # Rebuild index
        self.kb.fact_index.clear()
        for fact in self.kb.facts:
            self.kb.fact_index[fact.subject].add(fact)
            self.kb.fact_index[fact.object].add(fact)

        logger.info(f"Loaded knowledge base from {path}")


if __name__ == "__main__":
    # Test Symbolic Reasoner
    logger.info("Testing Symbolic Reasoner...")

    reasoner = SymbolicReasoner()

    # Add facts
    reasoner.add_fact("cat", "is_a", "animal")
    reasoner.add_fact("animal", "is_a", "living_thing")
    reasoner.add_fact("dog", "is_a", "animal")
    reasoner.add_fact("cat", "similar_to", "dog")

    print(f"\nInitial facts: {len(reasoner.kb.facts)}")

    # Perform reasoning
    new_facts = reasoner.reason()

    print(f"Total facts after reasoning: {len(reasoner.kb.facts)}")
    print("\nDerived facts:")
    for fact in new_facts:
        print(f"  {fact.subject} {fact.predicate} {fact.object} (conf: {fact.confidence})")

    # Query
    print("\nQuery: What is a cat?")
    results = reasoner.kb.query(subject="cat", predicate="is_a")
    for fact in results:
        print(f"  {fact.subject} {fact.predicate} {fact.object}")

    # Get related entities
    print("\nEntities related to 'cat':")
    related = reasoner.kb.get_related_entities("cat", max_hops=2)
    print(f"  {related}")

    logger.info("\nSymbolic reasoner tests complete!")

