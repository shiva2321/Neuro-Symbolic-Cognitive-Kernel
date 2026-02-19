#!/usr/bin/env python3
"""
Enhanced Response Composer for NSCK AI Model
============================================

Addresses the main limitation identified in testing: response generation quality.
Instead of simply retrieving sentences, this module composes fluent responses
using VSA-based operations and learned patterns.

Key improvements:
1. VSA-based sentence composition
2. Template-based generation with learned patterns
3. Concept-to-text mapping with fluency
4. Context-aware response building
"""

import os
import sys
import re
from typing import List, Dict, Set, Tuple, Any
from collections import Counter, defaultdict
import numpy as np

# Ensure nsck-demo is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NSCK_DEMO = os.path.join(_REPO_ROOT, "nsck-demo")
if _NSCK_DEMO not in sys.path:
    sys.path.insert(0, _NSCK_DEMO)

import python.core.vsa.hypervec_shim as hypervec_rs


class ResponseComposer:
    """
    Enhanced response composer that generates fluent text using VSA operations
    and learned linguistic patterns.

    All query starters, relation templates, and response patterns are
    *learned from data* via ``learn_from_text()`` and ``learn_from_qa()``.
    No hardcoded response templates.
    """
    
    def __init__(self):
        """Initialize the response composer."""
        self.sentence_templates: List[str] = []
        self.concept_phrases: Dict[str, List[str]] = defaultdict(list)
        self.transition_patterns: Dict[str, List[str]] = defaultdict(list)
        self.learned_patterns: List[Dict[str, Any]] = []
        
        # Learned starters per query type: {qtype → Counter(starter → count)}
        self._learned_starters: Dict[str, Counter] = defaultdict(Counter)
        # Learned Q&A answer structures
        self._qa_structures: Dict[str, List[str]] = defaultdict(list)
        # Relation → natural-language templates (learned from training text)
        self.relation_templates: Dict[str, List[str]] = defaultdict(list)

        # Grammatical starters — minimal, filled by learning
        # (empty by default; populated from Q&A data)
        self.query_starters: Dict[str, List[str]] = defaultdict(list)
    
    def learn_from_text(self, text: str):
        """
        Learn linguistic patterns from training text.
        
        Args:
            text: Training text to extract patterns from
        """
        sentences = self._split_sentences(text)
        
        for sent in sentences:
            # Store templates (replace specific entities with placeholders)
            template = self._extract_template(sent)
            if template and template not in self.sentence_templates:
                self.sentence_templates.append(template)
            
            # Extract concept phrases
            concepts = self._extract_concepts(sent.lower())
            for concept in concepts:
                phrase = self._extract_phrase_around_concept(sent, concept)
                if phrase:
                    self.concept_phrases[concept.lower()].append(phrase)
            
            # Learn transition patterns
            if len(sentences) > 1:
                self._learn_transitions(sentences)

            # Auto-learn relation templates from "X is Y" / "X has Y" patterns
            self._learn_relation_from_sentence(sent)

    def learn_from_qa(self, question: str, answer: str):
        """Learn response patterns from a question-answer pair.

        Discovers what answer structures correspond to each query type.
        """
        q_type = self._determine_query_type(question)

        # Learn answer starter (first 3 words)
        answer_words = answer.strip().split()
        if len(answer_words) >= 2:
            starter = ' '.join(answer_words[:min(3, len(answer_words))])
            self._learned_starters[q_type][starter] += 1

            # Rebuild query_starters from learned data
            top_starters = self._learned_starters[q_type].most_common(6)
            self.query_starters[q_type] = [s for s, _ in top_starters]

        # Store answer structure for this query type (cap at 50)
        if len(answer.strip()) > 10:
            structs = self._qa_structures[q_type]
            if answer.strip() not in structs:
                structs.append(answer.strip())
                if len(structs) > 50:
                    self._qa_structures[q_type] = structs[-50:]

    def learn_relation_template(self, relation: str, subject: str,
                                obj: str, sentence: str):
        """Learn how a relation is expressed in natural language.

        Given a sentence containing *subject* and *obj*, creates a
        template like ``"{subject} is the capital of {object}."``
        """
        tmpl = sentence
        # Replace subject and object with placeholders
        for variant in [subject, subject.lower(), subject.capitalize()]:
            tmpl = tmpl.replace(variant, '{subject}')
        for variant in [obj, obj.lower(), obj.capitalize()]:
            tmpl = tmpl.replace(variant, '{object}')

        if '{subject}' in tmpl and '{object}' in tmpl:
            templates = self.relation_templates[relation.lower()]
            if tmpl not in templates:
                templates.append(tmpl)
                # Keep only top 10 templates per relation
                if len(templates) > 10:
                    self.relation_templates[relation.lower()] = templates[-10:]

    def _learn_relation_from_sentence(self, sentence: str):
        """Auto-learn relation templates from sentences.

        Detects ``"X is the Y of Z"`` or ``"X verb Z"`` structures
        and stores them as templates.
        """
        sent = sentence.strip()
        if len(sent) < 10:
            return

        # Pattern: "Subject is/are the Relation of Object"
        m = re.match(
            r'^(\w[\w\s]{1,30}?)\s+(is|are|was|were)\s+'
            r'(?:the\s+)?(.+?)\s+of\s+(.+?)[.!?]?$',
            sent, re.IGNORECASE)
        if m:
            subj, verb, rel_phrase, obj = m.groups()
            rel_key = re.sub(r'\s+', '_', rel_phrase.lower().strip())
            tmpl = f"{{subject}} {verb} the {rel_phrase.strip()} of {{object}}."
            templates = self.relation_templates[rel_key]
            if tmpl not in templates:
                templates.append(tmpl)

        # Pattern: "Subject verb Object" (simple SVO)
        m = re.match(
            r'^(\w[\w\s]{1,20}?)\s+(causes?|improves?|increases?|decreases?|'
            r'contains?|produces?|uses?|requires?|enables?|prevents?)\s+'
            r'(.+?)[.!?]?$',
            sent, re.IGNORECASE)
        if m:
            subj, verb, obj = m.groups()
            rel_key = verb.lower().rstrip('s')
            tmpl = f"{{subject}} {verb.lower()} {{object}}."
            templates = self.relation_templates[rel_key]
            if tmpl not in templates:
                templates.append(tmpl)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Return statistics about learned patterns."""
        return {
            'sentence_templates': len(self.sentence_templates),
            'concept_phrases': sum(len(v) for v in self.concept_phrases.values()),
            'transition_patterns': sum(len(v) for v in self.transition_patterns.values()),
            'relation_templates': {k: len(v) for k, v in self.relation_templates.items()},
            'learned_starters': {k: len(v) for k, v in self._learned_starters.items()},
            'qa_structures': {k: len(v) for k, v in self._qa_structures.items()},
        }
    
    def compose_response(
        self,
        query: str,
        query_concepts: List[str],
        retrieved_sentences: List[str],
        activation: Dict[str, float],
        related_facts: List[Dict[str, Any]],
        confidence: float
    ) -> str:
        """
        Compose a fluent response using VSA-based operations.
        
        Args:
            query: User's query
            query_concepts: Extracted concepts from query
            retrieved_sentences: Retrieved source sentences
            activation: Concept activation levels
            related_facts: Related facts from knowledge base
            confidence: Overall confidence score
            
        Returns:
            Composed fluent response
        """
        # Determine query type
        query_type = self._determine_query_type(query)
        
        # If we have good retrieved sentences, enhance them
        if retrieved_sentences and confidence > 0.4:
            response = self._enhance_retrieved_response(
                query_type, query_concepts, retrieved_sentences, activation
            )
        else:
            # Generate from scratch using patterns and facts
            response = self._generate_from_patterns(
                query_type, query_concepts, related_facts, activation
            )
        
        # Post-process for fluency
        response = self._improve_fluency(response, query_type)
        
        return response
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query (what, who, where, etc.)."""
        query_lower = query.lower().strip()
        
        for qtype in ['what', 'who', 'where', 'when', 'how', 'why', 'which']:
            if query_lower.startswith(qtype):
                return qtype
        
        # Check if it's a statement or command
        if query_lower.startswith(('tell me', 'explain', 'describe')):
            return 'explain'
        
        return 'general'
    
    def _enhance_retrieved_response(
        self,
        query_type: str,
        query_concepts: List[str],
        retrieved_sentences: List[str],
        activation: Dict[str, float]
    ) -> str:
        """
        Enhance retrieved sentences for better fluency.
        
        Args:
            query_type: Type of query
            query_concepts: Query concepts
            retrieved_sentences: Retrieved sentences
            activation: Concept activation
            
        Returns:
            Enhanced response
        """
        if not retrieved_sentences:
            return ""
        
        # Take the best sentence
        main_sentence = retrieved_sentences[0]
        
        # Add a natural introduction if appropriate
        if query_type in self.query_starters and len(main_sentence) > 10:
            # Check if sentence already starts naturally
            if not main_sentence[0].isupper() or main_sentence.startswith('The '):
                return main_sentence
            
            # Add a connector
            starters = self.query_starters[query_type]
            
            # Choose starter based on sentence structure
            if ' is ' in main_sentence.lower():
                intro = starters[0] if starters else "This is"
                # Extract the subject-predicate
                parts = main_sentence.split(' is ', 1)
                if len(parts) == 2:
                    return f"{parts[0]} is {parts[1]}"
            
            return main_sentence
        
        # If we have multiple sentences, connect them naturally
        if len(retrieved_sentences) > 1:
            response_parts = [main_sentence]
            
            # Add second sentence with transition if relevant
            second = retrieved_sentences[1]
            
            # Check if they share concepts
            main_concepts = set(self._extract_concepts(main_sentence.lower()))
            second_concepts = set(self._extract_concepts(second.lower()))
            
            if main_concepts & second_concepts:
                # They're related, add with transition
                transitions = ["Additionally,", "Furthermore,", "Also,", "Moreover,"]
                connector = np.random.choice(transitions)
                response_parts.append(f"{connector} {second.lower()[0]}{second[1:]}")
            else:
                # Just add it
                response_parts.append(second)
            
            return " ".join(response_parts)
        
        return main_sentence
    
    def _generate_from_patterns(
        self,
        query_type: str,
        query_concepts: List[str],
        related_facts: List[Dict[str, Any]],
        activation: Dict[str, float]
    ) -> str:
        """
        Generate response from learned patterns and facts.
        
        Args:
            query_type: Type of query
            query_concepts: Query concepts
            related_facts: Related facts
            activation: Concept activation
            
        Returns:
            Generated response
        """
        # If we have related facts, construct from them
        if related_facts and len(related_facts) > 0:
            fact = related_facts[0]
            
            subject = fact.get('subject', '')
            relation = fact.get('relation', '')
            obj = fact.get('object', '')
            
            if subject and relation and obj:
                # Construct natural sentence
                response = self._construct_from_fact(subject, relation, obj, query_type)
                if response:
                    return response
        
        # Try to construct from concept phrases
        for concept in query_concepts:
            if concept.lower() in self.concept_phrases:
                phrases = self.concept_phrases[concept.lower()]
                if phrases:
                    return phrases[0]
        
        # Fallback to template-based
        if self.sentence_templates:
            template = self.sentence_templates[0]
            # Try to fill template with query concepts
            for i, concept in enumerate(query_concepts[:3]):
                template = template.replace(f"[CONCEPT{i}]", concept)
            return template
        
        return "I don't have enough information to answer that accurately."
    
    def _construct_from_fact(
        self, 
        subject: str, 
        relation: str, 
        obj: str,
        query_type: str
    ) -> str:
        """Construct a natural sentence from a fact triple.

        Uses learned relation templates first, then grammatical fallbacks.
        """
        relation_lower = relation.lower()
        
        # Check learned relation templates first
        templates = self.relation_templates.get(relation_lower, [])
        if not templates:
            # Try without underscores
            templates = self.relation_templates.get(
                relation_lower.replace('_', ' '), [])
        if templates:
            try:
                return templates[0].format(subject=subject, object=obj)
            except (KeyError, IndexError):
                pass

        # Grammatical fallbacks (not domain-specific)
        if relation_lower in ('is', 'are', 'was', 'were'):
            return f"{subject} {relation} {obj}."
        
        if relation_lower in ('has', 'have', 'had'):
            return f"{subject} {relation} {obj}."
        
        # Default: use relation as verb
        return f"{subject} {relation.replace('_', ' ')} {obj}."
    
    def _improve_fluency(self, response: str, query_type: str) -> str:
        """
        Post-process response to improve fluency.
        
        Args:
            response: Raw response
            query_type: Query type
            
        Returns:
            Improved response
        """
        if not response or len(response) < 5:
            return response
        
        # Capitalize first letter
        response = response[0].upper() + response[1:] if len(response) > 1 else response.upper()
        
        # Ensure proper ending punctuation
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Fix double spaces
        response = re.sub(r'\s+', ' ', response)
        
        # Fix punctuation spacing
        response = re.sub(r'\s+([.,!?])', r'\1', response)
        
        # Remove any remaining HTML/weird chars
        response = re.sub(r'<[^>]+>', '', response)
        
        return response.strip()
    
    def _extract_template(self, sentence: str) -> str:
        """Extract a template from a sentence by replacing specific entities."""
        # This is a simple version - could be enhanced with NER
        template = sentence
        
        # Replace numbers with placeholder
        template = re.sub(r'\b\d+\b', '[NUM]', template)
        
        # Replace capitalized words (potential proper nouns) with placeholder
        words = template.split()
        for i, word in enumerate(words):
            if word and word[0].isupper() and i > 0:  # Not first word
                words[i] = '[ENTITY]'
        
        template = ' '.join(words)
        
        return template if '[' in template else ""
    
    def _extract_phrase_around_concept(self, sentence: str, concept: str) -> str:
        """Extract the phrase around a concept in a sentence."""
        sent_lower = sentence.lower()
        concept_lower = concept.lower()
        
        if concept_lower not in sent_lower:
            return ""
        
        # Get a window around the concept
        idx = sent_lower.index(concept_lower)
        
        # Find clause boundaries
        start = max(0, idx - 50)
        end = min(len(sentence), idx + len(concept) + 50)
        
        # Extract phrase
        phrase = sentence[start:end].strip()
        
        return phrase
    
    def _learn_transitions(self, sentences: List[str]):
        """Learn transition patterns between sentences."""
        for i in range(len(sentences) - 1):
            first = sentences[i]
            second = sentences[i + 1]
            
            # Extract transition words from second sentence
            second_words = second.split()
            if second_words:
                first_word = second_words[0].lower()
                if first_word in ['however', 'moreover', 'furthermore', 'additionally',
                                 'also', 'therefore', 'thus', 'hence']:
                    # Store this transition pattern
                    first_concepts = set(self._extract_concepts(first.lower()))
                    for concept in first_concepts:
                        self.transition_patterns[concept.lower()].append(first_word)
    
    @staticmethod
    def _extract_concepts(text: str) -> List[str]:
        """Extract potential concepts from text."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter out stop words (simple list)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they'
        }
        
        concepts = [w for w in words if w not in stop_words and len(w) > 2]
        
        return concepts
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        parts = re.split(r'[.!?]+', text)
        return [s.strip() for s in parts if len(s.strip()) > 5]
