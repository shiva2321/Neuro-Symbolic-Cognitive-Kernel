"""
NCGN Ingestion Module - NLP-based Document Reader

Uses spaCy for dependency parsing to extract triples from text.
This is a SENSOR, not an AUTHORITY - all extracted triples start
with LOW confidence (0.2).

Key Principle: spaCy parses syntax, it does not determine truth.

Features:
- Sentence segmentation
- Subject-Verb-Object extraction
- Adjective/Property extraction
- Source tracking for traceability
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

# spaCy is optional - graceful degradation
try:
    import spacy
    from spacy.tokens import Doc, Token
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("Warning: spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")


class RelationType(Enum):
    """Types of relations extracted from text."""
    ACTION = "action"           # Subject performs action on object
    PROPERTY = "property"       # Subject has property
    IS_A = "is_a"              # Subject is a type of object
    ASSOCIATES = "associates"   # Generic association
    NEGATION = "negation"       # Negated relation


@dataclass
class Triple:
    """
    Extracted triple with source tracking.
    
    CRITICAL: Confidence starts LOW (0.2) because this is sensor output.
    System 2 must validate before these become trusted knowledge.
    """
    subject: str
    predicate: str
    object: str
    relation_type: RelationType = RelationType.ASSOCIATES
    confidence: float = 0.2  # LOW by default - spaCy is sensor, not authority
    source_line: int = 0
    source_text: str = ""
    flagged: bool = False     # True if schema-check failed
    flag_reason: str = ""     # Why it was flagged
    negated: bool = False     # True if relation is negated (e.g., "does not eat")
    
    def __repr__(self) -> str:
        neg = "¬" if self.negated else ""
        flag = " ⚠️" if self.flagged else ""
        return f"Triple({self.subject} --[{neg}{self.predicate}]--> {self.object}, C={self.confidence:.2f}){flag}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'relation_type': self.relation_type.value,
            'confidence': self.confidence,
            'source_line': self.source_line,
            'source_text': self.source_text,
            'flagged': self.flagged,
            'flag_reason': self.flag_reason,
            'negated': self.negated
        }


class DocumentReader:
    """
    NLP-based document reader using spaCy.
    
    Extracts triples from text without injecting them into the graph.
    All extraction goes to StagingBuffer for schema-checking first.
    """
    
    # Default confidence for extracted triples (LOW - sensor output)
    DEFAULT_CONFIDENCE = 0.2
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize with spaCy model.
        
        Args:
            model_name: spaCy model to use. Default is small English model.
        """
        self.nlp = None
        self.model_name = model_name
        
        if HAS_SPACY:
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                print(f"Warning: spaCy model '{model_name}' not found.")
                print(f"Run: python -m spacy download {model_name}")
    
    def is_ready(self) -> bool:
        """Check if NLP is ready."""
        return self.nlp is not None
    
    def parse_text(self, text: str) -> List[Triple]:
        """
        Parse text and extract triples.
        
        Algorithm:
        1. Sentence segmentation
        2. For each sentence:
           a. Find ROOT verb
           b. Extract nsubj (subject) and dobj (direct object)
           c. Extract adjectives as property relations
           d. Detect negation
        
        Returns:
            List of Triple with LOW confidence
        """
        if not self.is_ready():
            return self._fallback_parse(text)
        
        doc = self.nlp(text)
        triples = []
        
        for sent_idx, sent in enumerate(doc.sents):
            sent_triples = self._extract_sentence_triples(sent, sent_idx + 1)
            triples.extend(sent_triples)
        
        return triples
    
    def _extract_sentence_triples(self, sent, line_num: int) -> List[Triple]:
        """Extract triples from a single sentence."""
        triples = []
        sent_text = sent.text.strip()
        
        # Find the main verb (ROOT)
        root = None
        for token in sent:
            if token.dep_ == "ROOT":
                root = token
                break
        
        if not root:
            return triples
        
        # Check for negation
        is_negated = any(child.dep_ == "neg" for child in root.children)
        
        # Find subject and object
        subject = None
        obj = None
        
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject = self._get_compound_noun(child)
            elif child.dep_ in ("dobj", "pobj", "attr"):
                obj = self._get_compound_noun(child)
        
        # Also check prep phrases for objects
        if not obj:
            for child in root.children:
                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            obj = self._get_compound_noun(pobj)
                            break
        
        # Create main triple if we have subject and object
        if subject and obj:
            # Determine relation type
            rel_type = self._classify_relation(root.lemma_, subject, obj)
            
            triple = Triple(
                subject=subject.lower(),
                predicate=root.lemma_.lower(),
                object=obj.lower(),
                relation_type=rel_type,
                confidence=self.DEFAULT_CONFIDENCE,
                source_line=line_num,
                source_text=sent_text,
                negated=is_negated
            )
            triples.append(triple)
        
        # Extract adjective properties for subject
        if subject:
            for token in sent:
                if token.dep_ in ("amod", "acomp") and token.pos_ == "ADJ":
                    # Find what this adjective modifies
                    head = token.head
                    if head.text.lower() in subject.lower():
                        prop_triple = Triple(
                            subject=subject.lower(),
                            predicate="has_property",
                            object=token.lemma_.lower(),
                            relation_type=RelationType.PROPERTY,
                            confidence=self.DEFAULT_CONFIDENCE,
                            source_line=line_num,
                            source_text=sent_text,
                            negated=is_negated
                        )
                        triples.append(prop_triple)
        
        return triples
    
    def _get_compound_noun(self, token: 'Token') -> str:
        """Get compound noun phrase (e.g., 'robot dog' instead of just 'dog')."""
        compounds = []
        
        # Get compound modifiers
        for child in token.children:
            if child.dep_ == "compound":
                compounds.append(child.text)
        
        compounds.append(token.text)
        return " ".join(compounds)
    
    def _classify_relation(self, verb: str, subject: str, obj: str) -> RelationType:
        """Classify the relation type based on verb."""
        # is_a relations
        if verb in ("be", "is", "are", "was", "were"):
            return RelationType.IS_A
        
        # Action verbs
        if verb in ("eat", "eats", "run", "runs", "fly", "flies", "swim", "swims",
                    "walk", "walks", "bite", "bites", "chase", "chases"):
            return RelationType.ACTION
        
        # Property verbs
        if verb in ("have", "has", "had", "possess", "contain", "contains"):
            return RelationType.PROPERTY
        
        return RelationType.ASSOCIATES
    
    def _fallback_parse(self, text: str) -> List[Triple]:
        """
        Fallback parser when spaCy is not available.
        
        Uses simple heuristics to extract basic triples.
        """
        triples = []
        lines = text.split('.')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            words = line.lower().split()
            if len(words) >= 3:
                # Very simple SVO extraction
                subject = words[0]
                verb = words[1] if len(words) > 1 else "associates"
                obj = words[-1] if len(words) > 2 else ""
                
                if obj:
                    triple = Triple(
                        subject=subject,
                        predicate=verb,
                        object=obj,
                        relation_type=RelationType.ASSOCIATES,
                        confidence=self.DEFAULT_CONFIDENCE * 0.5,  # Even lower for fallback
                        source_line=line_num,
                        source_text=line + "."
                    )
                    triples.append(triple)
        
        return triples
    
    def parse_file(self, file_path: str, encoding: str = 'utf-8') -> List[Triple]:
        """
        Parse a text file and extract triples.
        
        Args:
            file_path: Path to text file
            encoding: File encoding (default UTF-8)
        
        Returns:
            List of extracted triples
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            return self.parse_text(text)
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return []
        except UnicodeDecodeError:
            print(f"Error: Could not decode file. Try different encoding.")
            return []


def text_to_triples(text: str, model: str = "en_core_web_sm") -> List[Triple]:
    """
    Convenience function to extract triples from text.
    
    Args:
        text: Raw text to parse
        model: spaCy model to use
    
    Returns:
        List of Triple objects with LOW confidence (sensor output)
    
    Example:
        >>> triples = text_to_triples("The hungry dog eats the metal.")
        >>> for t in triples:
        ...     print(t)
        Triple(hungry dog --[eat]--> metal, C=0.20)
        Triple(hungry dog --[has_property]--> hungry, C=0.20)
    """
    reader = DocumentReader(model)
    return reader.parse_text(text)


# CLI for testing
if __name__ == "__main__":
    test_sentences = [
        "The hungry dog eats the metal.",
        "Birds fly in the sky.",
        "Penguins are birds.",
        "Penguins do not fly.",
        "The robot dog eats metal parts.",
        "Dogs are animals.",
        "Metal is not edible.",
    ]
    
    print("=" * 60)
    print("NCGN Document Reader - Triple Extraction Test")
    print("=" * 60)
    
    reader = DocumentReader()
    
    if not reader.is_ready():
        print("\n⚠️ spaCy not available. Using fallback parser.")
        print("Install with: pip install spacy && python -m spacy download en_core_web_sm")
    
    for sentence in test_sentences:
        print(f"\n📝 Input: \"{sentence}\"")
        triples = reader.parse_text(sentence)
        
        if triples:
            print("   Extracted triples:")
            for t in triples:
                print(f"   • {t}")
        else:
            print("   (No triples extracted)")
    
    print("\n" + "=" * 60)
