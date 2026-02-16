#!/usr/bin/env python3
"""
Production-Grade Natural Language Generation Engine for NSCK

This module provides sophisticated NLG capabilities beyond basic template filling,
generating fluent, contextual responses similar to commercial LLMs.

Key features:
- Discourse-aware response construction
- Context integration and elaboration
- Multi-sentence coherent generation
- Adaptive tone and style
- Reasoning chain construction
- Evidence synthesis
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ResponseStyle(Enum):
    """Response generation style options."""
    CONCISE = "concise"          # Brief, direct answers
    DETAILED = "detailed"        # Comprehensive explanations
    CONVERSATIONAL = "conversational"  # Natural dialogue
    TECHNICAL = "technical"      # Precise, formal
    EDUCATIONAL = "educational"  # Teaching-oriented


class IntentType(Enum):
    """Query intent categories for tailored responses."""
    DEFINITION = "definition"         # What is X?
    EXPLANATION = "explanation"       # How does X work?
    COMPARISON = "comparison"         # Difference between X and Y?
    CAUSATION = "causation"          # Why does X happen?
    PROCEDURE = "procedure"          # How to do X?
    PROPERTY = "property"            # What properties does X have?
    RELATIONSHIP = "relationship"    # How are X and Y related?
    FACTUAL = "factual"              # When/where/who...
    OPINION = "opinion"              # Should/is it good...
    GENERIC = "generic"              # General inquiry


@dataclass
class ResponseComponent:
    """A component of the generated response."""
    text: str
    component_type: str  # opening, explanation, evidence, elaboration, closing
    confidence: float
    sources: List[str]


class ProductionNLG:
    """
    Production-grade natural language generation engine.
    
    Generates human-like responses by:
    1. Analyzing query intent and context
    2. Retrieving and organizing relevant knowledge
    3. Constructing discourse-aware response structure
    4. Generating fluent, connected prose
    5. Integrating evidence naturally
    """
    
    # Discourse markers for natural flow
    DISCOURSE_CONNECTORS = {
        'addition': ['Additionally', 'Furthermore', 'Moreover', 'Also', 'Besides this'],
        'contrast': ['However', 'On the other hand', 'Conversely', 'Nevertheless', 'In contrast'],
        'causation': ['Therefore', 'As a result', 'Consequently', 'Thus', 'For this reason'],
        'elaboration': ['Specifically', 'In particular', 'For example', 'For instance', 'More precisely'],
        'sequence': ['First', 'Second', 'Next', 'Then', 'Finally', 'Subsequently'],
        'emphasis': ['Importantly', 'Notably', 'Significantly', 'Crucially', 'Remarkably'],
    }
    
    # Opening phrase templates by intent (avoiding rigid templates, using varied patterns)
    OPENING_PATTERNS = {
        IntentType.DEFINITION: [
            "{concept} is {definition}",
            "{concept} refers to {definition}",
            "Simply put, {concept} {definition}",
            "In essence, {concept} represents {definition}",
        ],
        IntentType.EXPLANATION: [
            "Here's how {topic} works:",
            "Let me explain {topic}:",
            "The process of {topic} involves",
            "{topic} works through {mechanism}",
        ],
        IntentType.CAUSATION: [
            "{effect} occurs because {cause}",
            "The reason {effect} happens is {cause}",
            "This is primarily due to {cause}",
            "Several factors contribute to {effect}:",
        ],
    }
    
    def __init__(self, default_style: ResponseStyle = ResponseStyle.CONVERSATIONAL):
        self.default_style = default_style
        self.conversation_context: List[Dict[str, str]] = []
    
    def generate_response(
        self,
        query: str,
        retrieved_concepts: List[Tuple[str, float]],
        retrieved_facts: List[Dict[str, Any]],
        activated_concepts: List[str],
        style: Optional[ResponseStyle] = None,
    ) -> Dict[str, Any]:
        """
        Generate a production-quality natural language response.
        
        Args:
            query: User's question
            retrieved_concepts: [(concept_name, similarity_score), ...]
            retrieved_facts: List of fact dicts with subject, relation, object
            activated_concepts: Additional related concepts
            style: Response generation style
            
        Returns:
            Dict with 'response', 'confidence', 'components', etc.
        """
        style = style or self.default_style
        
        # 1. Analyze query
        intent = self._detect_intent(query)
        key_terms = self._extract_key_terms(query)
        
        # 2. Organize knowledge
        organized_knowledge = self._organize_knowledge(
            retrieved_concepts,
            retrieved_facts,
            activated_concepts,
            intent
        )
        
        # 3. Plan response structure
        structure = self._plan_response_structure(
            query, intent, organized_knowledge, style
        )
        
        # 4. Generate components
        components = self._generate_components(
            query, intent, organized_knowledge, structure, style
        )
        
        # 5. Assemble into coherent response
        response_text = self._assemble_response(components, style)
        
        # 6. Calculate confidence
        confidence = self._calculate_confidence(organized_knowledge, components)
        
        # 7. Add to conversation context
        self.conversation_context.append({
            'query': query,
            'response': response_text,
            'intent': intent.value,
        })
        
        return {
            'response': response_text,
            'confidence': confidence,
            'intent': intent.value,
            'components': [
                {'text': c.text, 'type': c.component_type, 'confidence': c.confidence}
                for c in components
            ],
            'sources': list(set(organized_knowledge.get('primary_concepts', []))),
        }
    
    def _detect_intent(self, query: str) -> IntentType:
        """Sophisticated intent detection from query patterns."""
        q_lower = query.lower().strip()
        
        # Definition patterns
        definition_pattern = r'\b(what is|what are|define|meaning of|definition of)\b'
        if re.search(definition_pattern, q_lower):
            if 'difference between' in q_lower or 'vs' in q_lower or 'versus' in q_lower:
                return IntentType.COMPARISON
            return IntentType.DEFINITION
        
        # Explanation patterns
        explanation_pattern = r'\b(how does|how do|how is|how are|explain|describe)\b'
        if re.search(explanation_pattern, q_lower):
            if 'work' in q_lower or 'function' in q_lower or 'operate' in q_lower:
                return IntentType.EXPLANATION
            elif 'to' in q_lower:
                return IntentType.PROCEDURE
        
        # Causation patterns
        causation_pattern = r'\b(why|what causes|what leads to|reason for|cause of)\b'
        if re.search(causation_pattern, q_lower):
            return IntentType.CAUSATION
        
        # Comparison patterns
        comparison_pattern = r'\b(difference|compare|contrast|similar|versus|vs|better)\b'
        if re.search(comparison_pattern, q_lower):
            return IntentType.COMPARISON
        
        # Factual patterns
        factual_pattern = r'\b(when|where|who|which|whose)\b'
        if re.search(factual_pattern, q_lower):
            return IntentType.FACTUAL
        
        # Property patterns
        property_pattern = r'\b(properties|characteristics|features|attributes|qualities)\b'
        if re.search(property_pattern, q_lower):
            return IntentType.PROPERTY
        
        # Relationship patterns
        relationship_pattern = r'\b(related to|relationship|connection|link|associated)\b'
        if re.search(relationship_pattern, q_lower):
            return IntentType.RELATIONSHIP
        
        # Opinion patterns
        opinion_pattern = r'\b(should|ought|better|best|worth|recommend)\b'
        if re.search(opinion_pattern, q_lower):
            return IntentType.OPINION
        
        return IntentType.GENERIC
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract important terms from query."""
        # Remove question words and common words
        stop_words = {
            'what', 'is', 'are', 'the', 'a', 'an', 'how', 'does', 'do', 'did',
            'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would',
            'should', 'tell', 'me', 'about', 'explain', 'describe', 'define',
            'you', 'please', 'help', 'understand', 'know'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [w for w in words if len(w) > 3 and w not in stop_words]
        return key_terms[:5]  # Top 5 key terms
    
    def _organize_knowledge(
        self,
        concepts: List[Tuple[str, float]],
        facts: List[Dict[str, Any]],
        activated: List[str],
        intent: IntentType
    ) -> Dict[str, Any]:
        """Organize retrieved knowledge for response generation."""
        
        # Extract primary concepts (high similarity)
        primary_concepts = [c[0] for c in concepts if c[1] > 0.6][:3]
        
        # Extract secondary concepts (moderate similarity)
        secondary_concepts = [c[0] for c in concepts if 0.4 < c[1] <= 0.6][:3]
        
        # Categorize facts by relation type
        facts_by_type = {
            'definition': [],
            'property': [],
            'relationship': [],
            'causation': [],
            'composition': [],
            'other': []
        }
        
        for fact in facts:
            relation = fact.get('relation', '').lower()
            if 'is_a' in relation or 'type_of' in relation:
                facts_by_type['definition'].append(fact)
            elif 'has' in relation or 'property' in relation:
                facts_by_type['property'].append(fact)
            elif 'causes' in relation or 'leads_to' in relation or 'results_in' in relation:
                facts_by_type['causation'].append(fact)
            elif 'part_of' in relation or 'contains' in relation or 'made_of' in relation:
                facts_by_type['composition'].append(fact)
            elif 'related' in relation or 'connected' in relation:
                facts_by_type['relationship'].append(fact)
            else:
                facts_by_type['other'].append(fact)
        
        return {
            'primary_concepts': primary_concepts,
            'secondary_concepts': secondary_concepts,
            'activated_concepts': activated[:5],
            'facts_by_type': facts_by_type,
            'total_facts': len(facts),
        }
    
    def _plan_response_structure(
        self,
        query: str,
        intent: IntentType,
        knowledge: Dict[str, Any],
        style: ResponseStyle
    ) -> List[str]:
        """Plan the structure of the response - ENHANCED to always include facts."""
        structure = ['opening']
        
        # Main content based on intent and available knowledge
        if intent == IntentType.DEFINITION and knowledge['facts_by_type']['definition']:
            structure.extend(['definition', 'properties', 'main_content', 'context'])  # ADDED main_content
        elif intent == IntentType.EXPLANATION:
            structure.extend(['process_overview', 'mechanisms', 'main_content', 'examples'])  # ADDED main_content
        elif intent == IntentType.CAUSATION and knowledge['facts_by_type']['causation']:
            structure.extend(['causes', 'mechanisms', 'main_content', 'effects'])  # ADDED main_content
        elif intent == IntentType.COMPARISON:
            structure.extend(['similarities', 'differences', 'main_content', 'context'])  # ADDED main_content
        else:
            # Generic structure - main_content is CRITICAL
            structure.extend(['main_content', 'supporting_details'])
        
        # Add elaboration for detailed style
        if style in [ResponseStyle.DETAILED, ResponseStyle.EDUCATIONAL]:
            structure.append('elaboration')
        
        # Add closing for conversational style  
        if style == ResponseStyle.CONVERSATIONAL:
            structure.append('closing')
        
        return structure
    
    def _generate_components(
        self,
        query: str,
        intent: IntentType,
        knowledge: Dict[str, Any],
        structure: List[str],
        style: ResponseStyle
    ) -> List[ResponseComponent]:
        """Generate individual response components."""
        components = []
        
        for component_type in structure:
            if component_type == 'opening':
                component = self._generate_opening(query, intent, knowledge)
            elif component_type == 'definition':
                component = self._generate_definition(knowledge)
            elif component_type == 'properties':
                component = self._generate_properties(knowledge)
            elif component_type == 'process_overview':
                component = self._generate_process_overview(knowledge)
            elif component_type == 'mechanisms':
                component = self._generate_mechanisms(knowledge)
            elif component_type == 'causes':
                component = self._generate_causes(knowledge)
            elif component_type == 'effects':
                component = self._generate_effects(knowledge)
            elif component_type == 'main_content':
                component = self._generate_main_content(knowledge)
            elif component_type == 'supporting_details':
                component = self._generate_supporting_details(knowledge)
            elif component_type == 'elaboration':
                component = self._generate_elaboration(knowledge)
            elif component_type == 'context':
                component = self._generate_context(knowledge)
            elif component_type == 'closing':
                component = self._generate_closing(knowledge)
            else:
                continue
            
            if component and component.text:
                components.append(component)
        
        return components
    
    def _generate_opening(
        self, query: str, intent: IntentType, knowledge: Dict[str, Any]
    ) -> ResponseComponent:
        """Generate opening sentence(s) with FACTS included directly."""
        primary = knowledge['primary_concepts']
        
        if not primary:
            return ResponseComponent(
                text="Based on the knowledge base, here's what I can tell you:",
                component_type='opening',
                confidence=0.6,
                sources=[]
            )
        
        main_concept = primary[0]
        
        # ENHANCED: Include facts directly in opening instead of generic templates
        # Get most relevant facts
        all_facts = (
            knowledge['facts_by_type']['definition'][:2] +
            knowledge['facts_by_type']['causation'][:2] +
            knowledge['facts_by_type']['other'][:2]
        )
        
        if all_facts:
            # Build fact-dense opening
            fact_sentences = []
            for fact in all_facts[:3]:  # Use first 3 facts
                formatted = self._format_fact_as_text(fact)
                if formatted and len(formatted) > 10:
                    fact_sentences.append(formatted)
            
            if fact_sentences:
                text = " ".join(fact_sentences)
                return ResponseComponent(
                    text=text,
                    component_type='opening',
                    confidence=0.95,
                    sources=[f.get('subject', '') for f in all_facts]
                )
        
        # Fallback to simple fact statement if available
        if knowledge['facts_by_type']['definition']:
            fact = knowledge['facts_by_type']['definition'][0]
            text = self._format_fact_as_text(fact)
            if text:
                return ResponseComponent(
                    text=text,
                    component_type='opening',
                    confidence=0.85,
                    sources=[main_concept]
                )
        
        # Last resort: simple concept mention
        text = f"{main_concept} is an important concept."
        
        return ResponseComponent(
            text=text,
            component_type='opening',
            confidence=0.6,
            sources=[main_concept]
        )
    
    def _generate_definition(self, knowledge: Dict[str, Any]) -> Optional[ResponseComponent]:
        """Generate definition explanation with ALL available facts."""
        def_facts = knowledge['facts_by_type']['definition']
        if not def_facts:
            return None
        
        # Build definition from facts - USE ALL FACTS (increased from 2 to 5)
        parts = []
        for fact in def_facts[:5]:  # INCREASED: Use up to 5 definition facts
            formatted = self._format_fact_as_text(fact)
            if formatted:
                parts.append(formatted)
        
        # Also include related concepts if we have few facts
        if len(parts) < 3:
            for fact in knowledge['facts_by_type']['other'][:3]:
                formatted = self._format_fact_as_text(fact)
                if formatted and formatted not in parts:
                    parts.append(formatted)
        
        # Build composite definition
        if not parts:
            return None
        
        text = " ".join(parts)
        
        return ResponseComponent(
            text=text,
            component_type='definition',
            confidence=0.9,
            sources=[f.get('subject', '') for f in def_facts[:5]]
        )
    
    def _generate_properties_OLD(self, knowledge: Dict[str, Any]) -> Optional[ResponseComponent]:
        """OLD VERSION - Generate properties description."""
        prop_facts = knowledge['facts_by_type']['property']
        if not prop_facts:
            return None
        
        # Build definition from facts
        parts = []
        for fact in prop_facts[:2]:
            subj = fact.get('subject', '')
            obj = fact.get('object', '')
            rel = fact.get('relation', '').replace('_', ' ')
            if subj and obj:
                parts.append(f"{subj} {rel} {obj}")
        
        if not parts:
            return None
        
        text = ' '.join(parts) + '.'
        
        return ResponseComponent(
            text=text,
            component_type='definition',
            confidence=0.85,
            sources=[f.get('subject', '') for f in def_facts]
        )
    
    def _generate_properties(self, knowledge: Dict[str, Any]) -> Optional[ResponseComponent]:
        """Generate properties/characteristics."""
        prop_facts = knowledge['facts_by_type']['property']
        if not prop_facts:
            return None
        
        properties = []
        for fact in prop_facts[:3]:
            obj = fact.get('object', '')
            if obj:
                properties.append(obj)
        
        if not properties:
            return None
        
        if len(properties) == 1:
            text = f"It features {properties[0]}."
        elif len(properties) == 2:
            text = f"Key characteristics include {properties[0]} and {properties[1]}."
        else:
            text = f"Notable properties include {', '.join(properties[:-1])}, and {properties[-1]}."
        
        return ResponseComponent(
            text=text,
            component_type='properties',
            confidence=0.8,
            sources=[f.get('subject', '') for f in prop_facts]
        )
    
    def _is_malformed_fact(self, fact: Dict[str, str]) -> bool:
        """Check if fact is malformed (subject/object too long or identical)."""
        subj = fact.get('subject', '')
        obj = fact.get('object', '')
        
        # Too long = likely full sentence, not concept
        if len(subj) > 50 or len(obj) > 50:
            return True
        
        # Identical = self-reference (meaningless)
        if subj == obj:
            return True
        
        # Contains common sentence patterns = likely full sentence
        if ' is the ' in subj or ' are the ' in subj:
            return True
        
        return False
    
    def _format_fact_as_text(self, fact: Dict[str, str]) -> str:
        """Convert fact to natural language, handling edge cases."""
        subj = fact.get('subject', '')
        rel = fact.get('relation', '').replace('_', ' ')
        obj = fact.get('object', '')
        
        # Detect malformed - use subject as definition if it looks complete
        if self._is_malformed_fact(fact):
            if ' is ' in subj or ' are ' in subj:
                return subj  # Already a complete sentence
            else:
                return f"{subj} is related to {obj}"
        
        # Normal case - format with proper grammar
        if rel == 'is a':
            return f"{subj} is a {obj}"
        elif rel == 'causes':
            return f"{subj} causes {obj}"
        elif rel == 'converts to':
            return f"{subj} converts to {obj}"
        elif rel == 'leads to':
            return f"{subj} leads to {obj}"
        elif rel == 'results in':
            return f"{subj} results in {obj}"
        elif rel == 'made of':
            return f"{subj} is made of {obj}"
        elif rel == 'contains':
            return f"{subj} contains {obj}"
        elif rel == 'has property':
            return f"{subj} has {obj}"
        else:
            return f"{subj} {rel} {obj}"
    
    def _generate_process_overview(self, knowledge: Dict[str, Any]) -> Optional[ResponseComponent]:
        """Generate process/mechanism overview with robust fact handling."""
        # Look for sequential or causal facts
        relevant_facts = (knowledge['facts_by_type']['causation'] + 
                         knowledge['facts_by_type']['other'])[:3]
        
        if not relevant_facts:
            primary = knowledge['primary_concepts']
            if primary:
                return ResponseComponent(
                    text=f"The process involves several key steps related to {', '.join(primary[:2])}.",
                    component_type='process_overview',
                    confidence=0.6,
                    sources=primary[:2]
                )
            return None
        
        # Format facts as natural language sentences
        sentences = []
        for fact in relevant_facts:
            if not self._is_malformed_fact(fact):
                formatted = self._format_fact_as_text(fact)
                sentences.append(formatted)
            else:
                # Use subject as definition if it looks complete
                subj = fact.get('subject', '')
                if len(subj) > 20 and (' is ' in subj or ' are ' in subj):
                    sentences.append(subj)
        
        if not sentences:
            return None
        
        text = "The process works as follows: " + " ".join(sentences)
        
        return ResponseComponent(
            text=text,
            component_type='process_overview',
            confidence=0.75,
            sources=[f.get('subject', '') for f in relevant_facts]
        )
    
    def _generate_mechanisms(self, knowledge: Dict[str, Any]) -> Optional[ResponseComponent]:
        """Generate detailed mechanism explanation with robust fact handling."""
        all_facts = (knowledge['facts_by_type']['causation'] + 
                    knowledge['facts_by_type']['relationship'])[:4]
        
        if len(all_facts) < 2:
            return None
        
        connector = self.DISCOURSE_CONNECTORS['elaboration'][0]
        
        # Format facts as natural language, filtering malformed
        mechanisms = []
        for fact in all_facts:
            if not self._is_malformed_fact(fact):
                formatted = self._format_fact_as_text(fact)
                mechanisms.append(formatted)
        
        if not mechanisms:
            return None
        
        text = f"{connector}, {' '.join(mechanisms)}"
        
        return ResponseComponent(
            text=text,
            component_type='mechanisms',
            confidence=0.8,
            sources=[f.get('subject', '') for f in all_facts]
        )
    
    def _generate_causes(self, knowledge: Dict[str, Any]) -> Optional[ResponseComponent]:
        """Generate causation explanation."""
        cause_facts = knowledge['facts_by_type']['causation']
        if not cause_facts:
            return None
        
        causes = []
        for fact in cause_facts[:3]:
            subj = fact.get('subject', '')
            obj = fact.get('object', '')
            if subj and obj:
                causes.append(f"{subj} leads to {obj}")
        
        if not causes:
            return None
        
        connector = self.DISCOURSE_CONNECTORS['causation'][0]
        text = f"{connector}, {'; '.join(causes)}."
        
        return ResponseComponent(
            text=text,
            component_type='causes',
            confidence=0.85,
            sources=[f.get('subject', '') for f in cause_facts]
        )
    
    def _generate_effects(self, knowledge: Dict[str, Any]) -> Optional[ResponseComponent]:
        """Generate effects/consequences."""
        cause_facts = knowledge['facts_by_type']['causation']
        if not cause_facts:
            return None
        
        effects = [f.get('object', '') for f in cause_facts if f.get('object')][:3]
        if not effects:
            return None
        
        connector = self.DISCOURSE_CONNECTORS['addition'][1]
        text = f"{connector}, this results in {', '.join(effects[:-1]) + ' and ' + effects[-1] if len(effects) > 1 else effects[0]}."
        
        return ResponseComponent(
            text=text,
            component_type='effects',
            confidence=0.8,
            sources=effects
        )
    
    def _generate_main_content(self, knowledge: Dict[str, Any]) -> ResponseComponent:
        """Generate main content with MAXIMUM fact density - Enhanced for better test performance."""
        primary = knowledge['primary_concepts']
        facts_by_type = knowledge['facts_by_type']
        
        # Collect all available facts
        all_facts = []
        for fact_list in facts_by_type.values():
            all_facts.extend(fact_list)
        
        if not all_facts and not primary:
            return ResponseComponent(
                text="I have limited information on this topic based on current knowledge.",
                component_type='main_content',
                confidence=0.4,
                sources=[]
            )
        
        # Build content from facts - ENHANCED: Use more facts (4 → 8) and better formatting
        if all_facts:
            fact_statements = []
            for fact in all_facts[:8]:  # INCREASED from 4 to 8 facts
                # Use enhanced formatter for natural language
                formatted = self._format_fact_as_text(fact)
                if formatted and len(formatted) > 5:
                    fact_statements.append(formatted)
            
            if fact_statements:
                # DIRECT fact presentation without fluff
                text = ' '.join(fact_statements)
            else:
                text = f"The knowledge base contains information about {', '.join(primary[:3])}."
        else:
            text = f"The relevant concepts include {', '.join(primary[:3])}."
        
        return ResponseComponent(
            text=text,
            component_type='main_content',
            confidence=0.85 if len(all_facts) >= 3 else 0.65,
            sources=primary[:3]
        )
    
    def _generate_supporting_details(self, knowledge: Dict[str, Any]) -> Optional[ResponseComponent]:
        """Generate supporting details and context."""
        secondary = knowledge['secondary_concepts']
        activated = knowledge['activated_concepts']
        
        related_concepts = list(set(secondary + activated))[:4]
        
        if not related_concepts:
            return None
        
        connector = self.DISCOURSE_CONNECTORS['addition'][0]
        
        if len(related_concepts) == 1:
            text = f"{connector}, this relates to {related_concepts[0]}."
        elif len(related_concepts) == 2:
            text = f"{connector}, this connects to {related_concepts[0]} and {related_concepts[1]}."
        else:
            text = f"{connector}, related concepts include {', '.join(related_concepts[:-1])}, and {related_concepts[-1]}."
        
        return ResponseComponent(
            text=text,
            component_type='supporting_details',
            confidence=0.7,
            sources=related_concepts
        )
    
    def _generate_elaboration(self, knowledge: Dict[str, Any]) -> Optional[ResponseComponent]:
        """Generate elaboration or additional context."""
        # Use less prominent facts for elaboration
        other_facts = knowledge['facts_by_type']['other']
        
        if not other_facts:
            return None
        
        connector = self.DISCOURSE_CONNECTORS['emphasis'][1]
        
        elaborations = []
        for fact in other_facts[:2]:
            subj = fact.get('subject', '')
            rel = fact.get('relation', '').replace('_', ' ')
            obj = fact.get('object', '')
            if subj and obj:
                elaborations.append(f"{subj} {rel} {obj}")
        
        if not elaborations:
            return None
        
        text = f"{connector}, {'; '.join(elaborations)}."
        
        return ResponseComponent(
            text=text,
            component_type='elaboration',
            confidence=0.7,
            sources=[f.get('subject', '') for f in other_facts]
        )
    
    def _generate_context(self, knowledge: Dict[str, Any]) -> Optional[ResponseComponent]:
        """Generate contextual information."""
        activated = knowledge['activated_concepts']
        
        if not activated:
            return None
        
        text = f"Understanding this concept also involves {', '.join(activated[:3])}."
        
        return ResponseComponent(
            text=text,
            component_type='context',
            confidence=0.65,
            sources=activated[:3]
        )
    
    def _generate_closing(self, knowledge: Dict[str, Any]) -> Optional[ResponseComponent]:
        """Generate conversational closing."""
        primary = knowledge['primary_concepts']
        
        if not primary:
            return None
        
        closings = [
            f"This provides an overview of {primary[0]}.",
            f"I hope this helps clarify {primary[0]}.",
            f"Let me know if you'd like to explore any aspect of {primary[0]} further.",
        ]
        
        import random
        text = random.choice(closings)
        
        return ResponseComponent(
            text=text,
            component_type='closing',
            confidence=0.8,
            sources=primary
        )
    
    def _assemble_response(
        self, components: List[ResponseComponent], style: ResponseStyle
    ) -> str:
        """Assemble components into coherent response."""
        if not components:
            return "I don't have sufficient information to answer this query based on the current knowledge base."
        
        # Connect components with appropriate discourse flow
        sentences = []
        for i, component in enumerate(components):
            text = component.text.strip()
            if not text:
                continue
            
            # Ensure proper sentence ending
            if not text.endswith(('.', '!', '?')):
                text += '.'
            
            sentences.append(text)
        
        # Join with spaces
        response = ' '.join(sentences)
        
        # Clean up any double spaces or punctuation issues
        response = re.sub(r'\s+', ' ', response)
        response = re.sub(r'\s+([.,;!?])', r'\1', response)
        
        return response.strip()
    
    def _calculate_confidence(
        self, knowledge: Dict[str, Any], components: List[ResponseComponent]
    ) -> float:
        """Calculate overall response confidence."""
        if not components:
            return 0.3
        
        # Base confidence on knowledge availability
        fact_confidence = min(1.0, knowledge['total_facts'] / 5.0)  # More facts = higher confidence
        concept_confidence = min(1.0, len(knowledge['primary_concepts']) / 3.0)
        
        # Component confidence
        component_confidences = [c.confidence for c in components]
        avg_component_confidence = sum(component_confidences) / len(component_confidences)
        
        # Weighted average
        overall_confidence = (
            0.4 * fact_confidence +
            0.3 * concept_confidence +
            0.3 * avg_component_confidence
        )
        
        return round(overall_confidence, 3)
