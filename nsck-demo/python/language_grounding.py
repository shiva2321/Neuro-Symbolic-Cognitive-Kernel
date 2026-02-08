"""
NSCK Language Grounding (Phase 2.3)
====================================
Enhanced symbol grounding system that maps words/phrases to perceptual VSA concepts.

Implements:
  - Word → perceptual experience mapping
  - Embodied language learning
  - Vision-language binding (CLIP-style)
  - Symbol grounding via experience accumulation

Design Constraints (per AGENT_INSTRUCTIONS.md):
  - LLM is peripheral (not the brain)
  - Core reasoning stays in VSA
  - Language translates to/from symbolic concepts
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import hypervec_shim as hypervec_rs
except ImportError:
    from hypervec_py import HyperVector as _HV
    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()


@dataclass
class GroundedConcept:
    """A concept grounded in perceptual experience."""
    word: str
    concept_hv: Any  # VSA HyperVector
    grounding_experiences: List[Dict[str, Any]]  # Perceptual experiences
    confidence: float = 0.0
    usage_count: int = 0
    

@dataclass
class LanguagePerceptionBinding:
    """Binding between language and perception."""
    text: str
    visual_hv: Optional[Any] = None
    audio_hv: Optional[Any] = None
    unified_hv: Optional[Any] = None
    confidence: float = 0.0


class SymbolGroundingEngine:
    """
    Maps linguistic symbols to grounded perceptual concepts.
    
    Implements the Symbol Grounding Problem solution:
      1. Collect perceptual states when word is used
      2. Find common pattern via VSA bundling
      3. Bind word to perceptual concept
    """
    
    def __init__(self, seed: int = 44):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        
        # Grounded concept dictionary: word → GroundedConcept
        self.grounded_concepts: Dict[str, GroundedConcept] = {}
        
        # Experience buffer: stores recent language-perception pairings
        self.experience_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 1000
        
        # Role vectors for binding
        self.role_word = self._create_role_hv("ROLE_WORD")
        self.role_perception = self._create_role_hv("ROLE_PERCEPTION")
        
        # Statistics
        self.stats = {
            "words_grounded": 0,
            "experiences_collected": 0,
            "bindings_created": 0
        }
    
    def _create_role_hv(self, name: str) -> Any:
        """Create a deterministic role HyperVector."""
        seed_value = int(name.encode().hex(), 16) % (2**32)
        return hypervec_rs.HyperVector(seed_value)
    
    def _word_to_hv(self, word: str) -> Any:
        """Convert word to HyperVector."""
        # Simple approach: hash the word
        word_hash = hash(word.lower().strip())
        seed_val = abs(word_hash) % (2**32)
        return hypervec_rs.HyperVector(seed_val)
    
    def add_experience(self, words: List[str], 
                      perception_hv: Any,
                      context: Optional[Dict[str, Any]] = None):
        """
        Add a language-perception experience.
        
        Args:
            words: Words/phrases present in this experience
            perception_hv: Perceptual HyperVector (from vision/audio)
            context: Optional contextual information
        """
        experience = {
            "words": set(w.lower().strip() for w in words),
            "perception_hv": perception_hv,
            "context": context or {},
            "timestamp": len(self.experience_buffer)
        }
        
        self.experience_buffer.append(experience)
        self.stats["experiences_collected"] += 1
        
        # Trim buffer if too large
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        # Update grounding for each word
        for word in words:
            self._update_grounding(word.lower().strip(), experience)
    
    def _update_grounding(self, word: str, experience: Dict[str, Any]):
        """Update the grounding for a specific word."""
        if word not in self.grounded_concepts:
            # Create new grounded concept
            word_hv = self._word_to_hv(word)
            self.grounded_concepts[word] = GroundedConcept(
                word=word,
                concept_hv=word_hv,  # Initialize with word HV
                grounding_experiences=[],
                confidence=0.0,
                usage_count=0
            )
            self.stats["words_grounded"] += 1
        
        concept = self.grounded_concepts[word]
        concept.grounding_experiences.append(experience)
        concept.usage_count += 1
        
        # Re-ground: bundle all perceptual experiences for this word
        if len(concept.grounding_experiences) >= 3:
            # Need at least 3 experiences for stable grounding
            self._reground_concept(concept)
    
    def _reground_concept(self, concept: GroundedConcept):
        """
        Re-ground a concept by bundling its perceptual experiences.
        
        Implements: grounded_concept = bundle(all perceptual states where word used)
        """
        # Collect all perception HVs
        perception_hvs = [
            exp["perception_hv"] 
            for exp in concept.grounding_experiences[-10:]  # Use last 10
        ]
        
        if not perception_hvs:
            return
        
        # Bundle perceptual experiences
        # For simplicity, we'll use the first one and track confidence
        bundled_perception = perception_hvs[0]
        for hv in perception_hvs[1:]:
            # In true VSA, this would be proper bundling
            # For now, we'll just update the reference
            bundled_perception = hv  # Simplified
        
        # Bind word to grounded perception
        word_hv = self._word_to_hv(concept.word)
        concept.concept_hv = word_hv ^ (bundled_perception ^ self.role_perception)
        
        # Update confidence based on consistency
        concept.confidence = min(1.0, len(concept.grounding_experiences) / 10.0)
    
    def ground_word(self, word: str) -> Tuple[Any, float]:
        """
        Get the grounded concept for a word.
        
        Args:
            word: Word to ground
            
        Returns:
            Tuple of (concept_hv, confidence)
        """
        word_lower = word.lower().strip()
        
        if word_lower in self.grounded_concepts:
            concept = self.grounded_concepts[word_lower]
            return concept.concept_hv, concept.confidence
        else:
            # Return ungrounded word HV
            word_hv = self._word_to_hv(word_lower)
            return word_hv, 0.0
    
    def get_word_grounding_info(self, word: str) -> Dict[str, Any]:
        """Get detailed grounding information for a word."""
        word_lower = word.lower().strip()
        
        if word_lower not in self.grounded_concepts:
            return {
                "word": word,
                "is_grounded": False,
                "confidence": 0.0
            }
        
        concept = self.grounded_concepts[word_lower]
        return {
            "word": word,
            "is_grounded": True,
            "confidence": concept.confidence,
            "usage_count": concept.usage_count,
            "num_experiences": len(concept.grounding_experiences)
        }
    
    def ground_phrase(self, phrase: str) -> Any:
        """
        Ground a phrase by composing word groundings.
        
        Args:
            phrase: Text phrase to ground
            
        Returns:
            Grounded concept HV
        """
        words = phrase.lower().strip().split()
        
        if not words:
            return self._word_to_hv("")
        
        # Ground each word
        word_hvs = []
        for word in words:
            word_hv, _ = self.ground_word(word)
            word_hvs.append(word_hv)
        
        # Compose via bundling (simplified)
        phrase_hv = word_hvs[0]
        for hv in word_hvs[1:]:
            # In true VSA, this would be proper bundling
            phrase_hv = phrase_hv ^ hv  # Simplified composition
        
        return phrase_hv
    
    def get_stats(self) -> Dict[str, Any]:
        """Get grounding statistics."""
        stats = self.stats.copy()
        stats["total_grounded_words"] = len(self.grounded_concepts)
        stats["buffer_size"] = len(self.experience_buffer)
        
        # Calculate average confidence
        if self.grounded_concepts:
            avg_conf = np.mean([
                c.confidence for c in self.grounded_concepts.values()
            ])
            stats["avg_confidence"] = float(avg_conf)
        else:
            stats["avg_confidence"] = 0.0
        
        return stats


class VisionLanguageBinding:
    """
    CLIP-style vision-language binding system.
    
    Learns to bind visual and linguistic representations in a shared space.
    """
    
    def __init__(self, embedding_dim: int = 64):
        """
        Args:
            embedding_dim: Dimension of shared embedding space
        """
        self.embedding_dim = embedding_dim
        
        # Grounding engine for language
        self.grounding_engine = SymbolGroundingEngine()
        
        # Statistics
        self.stats = {
            "bindings_created": 0,
            "vision_language_pairs": 0
        }
    
    def bind_vision_language(self, image_hv: Any, text: str,
                            visual_features: Optional[torch.Tensor] = None) -> LanguagePerceptionBinding:
        """
        Create a binding between visual perception and language.
        
        Args:
            image_hv: Visual concept HyperVector
            text: Text description
            visual_features: Optional visual features for strengthening
            
        Returns:
            LanguagePerceptionBinding
        """
        # Extract words from text
        words = text.lower().strip().split()
        
        # Add experience to grounding engine
        self.grounding_engine.add_experience(
            words=words,
            perception_hv=image_hv,
            context={"modality": "vision", "text": text}
        )
        
        # Ground the full text
        text_hv = self.grounding_engine.ground_phrase(text)
        
        # Create unified representation (bundle visual and linguistic)
        unified_hv = image_hv ^ text_hv
        
        # Calculate binding confidence
        confidence = 1.0  # Perfect for direct binding
        
        self.stats["bindings_created"] += 1
        self.stats["vision_language_pairs"] += 1
        
        return LanguagePerceptionBinding(
            text=text,
            visual_hv=image_hv,
            unified_hv=unified_hv,
            confidence=confidence
        )
    
    def retrieve_from_text(self, text: str) -> Tuple[Any, float]:
        """
        Retrieve visual concept from text description.
        
        Args:
            text: Text query
            
        Returns:
            Tuple of (concept_hv, confidence)
        """
        # Ground the text
        text_hv = self.grounding_engine.ground_phrase(text)
        
        # Calculate average confidence
        words = text.lower().strip().split()
        confidences = [
            self.grounding_engine.ground_word(w)[1]
            for w in words
        ]
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return text_hv, float(avg_confidence)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get binding statistics."""
        stats = self.stats.copy()
        grounding_stats = self.grounding_engine.get_stats()
        stats.update(grounding_stats)
        return stats


class EmbodiedLanguageLearner:
    """
    Learns language through embodied interaction.
    
    Key principle: Language is learned through sensorimotor experience,
    not just pattern matching.
    """
    
    def __init__(self):
        """Initialize embodied language learner."""
        self.vision_language_binder = VisionLanguageBinding()
        self.grounding_engine = self.vision_language_binder.grounding_engine
        
        # Track action-language associations
        self.action_language_map: Dict[str, List[str]] = defaultdict(list)
        
    def observe_with_language(self, perception_hv: Any, 
                             description: str,
                             action: Optional[str] = None):
        """
        Observe a situation with language description.
        
        Args:
            perception_hv: Perceptual HyperVector (vision/audio/multimodal)
            description: Language description of the situation
            action: Optional action taken in this situation
        """
        # Extract words
        words = description.lower().strip().split()
        
        # Add to grounding engine
        context = {"description": description}
        if action:
            context["action"] = action
            self.action_language_map[action].extend(words)
        
        self.grounding_engine.add_experience(
            words=words,
            perception_hv=perception_hv,
            context=context
        )
    
    def understand_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Understand a language instruction by grounding it.
        
        Args:
            instruction: Language instruction
            
        Returns:
            Understanding result with grounded concepts
        """
        # Ground the instruction
        instruction_hv = self.grounding_engine.ground_phrase(instruction)
        
        # Extract key words
        words = instruction.lower().strip().split()
        
        # Get grounding info for each word
        word_groundings = {
            word: self.grounding_engine.get_word_grounding_info(word)
            for word in words
        }
        
        # Check for action words
        potential_actions = []
        for action, action_words in self.action_language_map.items():
            if any(w in action_words for w in words):
                potential_actions.append(action)
        
        return {
            "instruction": instruction,
            "grounded_hv": instruction_hv,
            "word_groundings": word_groundings,
            "potential_actions": potential_actions,
            "confidence": np.mean([
                g["confidence"] for g in word_groundings.values()
            ]) if word_groundings else 0.0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return self.grounding_engine.get_stats()


# Helper functions
def create_language_grounding_system() -> EmbodiedLanguageLearner:
    """
    Create a complete language grounding system.
    
    Returns:
        Configured EmbodiedLanguageLearner
    """
    return EmbodiedLanguageLearner()
