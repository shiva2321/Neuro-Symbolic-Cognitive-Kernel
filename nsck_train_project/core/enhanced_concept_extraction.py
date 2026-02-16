#!/usr/bin/env python3
"""
Enhanced Concept Extraction for NSCK Text Learner

Improves concept quality by:
1. Extracting multi-word noun phrases (not just single words)
2. Better named entity recognition patterns
3. Domain-specific compound terms
4. Preserving technical terminology
"""

import re
from typing import List, Set, Tuple


class EnhancedConceptExtractor:
    """
    Production-grade concept extraction for text learning.
    
    Extracts meaningful multi-word concepts instead of fragmenting them.
    """
    
    # Common multi-word patterns in technical/scientific text
    COMPOUND_PATTERNS = [
        # Science & Technology
        r'\b(artificial intelligence|machine learning|deep learning|neural network|quantum computing|quantum mechanics)\b',
        r'\b(climate change|global warming|greenhouse gas|carbon dioxide|renewable energy)\b',
        r'\b(natural language|computer science|data science|information technology)\b',
        r'\b(solar system|black hole|event horizon|gravitational wave|dark matter|dark energy)\b',
        r'\b(immune system|nervous system|circulatory system|respiratory system)\b',
        r'\b(DNA molecule|RNA molecule|amino acid|protein synthesis|cell membrane)\b',
        
        # General noun phrases (Adj + Noun, Noun + Noun)
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b',  # Multi-word proper nouns
        
        # Technical terms with hyphens
        r'\b([a-z]+-[a-z]+(?:-[a-z]+)*)\b',
        
        # Acronyms and initialisms
        r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b',
    ]
    
    # Domain-specific compound terms
    DOMAIN_COMPOUNDS = {
        'artificial intelligence', 'machine learning', 'deep learning',
        'neural network', 'quantum mechanics', 'quantum computing',
        'climate change', 'global warming', 'greenhouse gas',
        'natural language', 'computer science', 'data science',
        'solar system', 'black hole', 'white hole', 'worm hole',
        'immune system', 'nervous system', 'digestive system',
        'ribonucleic acid', 'deoxyribonucleic acid', 
        'amino acid', 'fatty acid', 'nucleic acid',
        'evolutionary biology', 'molecular biology', 'cellular biology',
        'renewable energy', 'fossil fuel', 'nuclear energy',
        'gravitational wave', 'electromagnetic radiation',
        'tectonic plate', 'continental drift', 'plate tectonics',
        'natural selection', 'genetic variation', 'evolutionary pressure',
        'internet protocol', 'transmission control', 'domain name',
        'operating system', 'file system', 'database management',
        'electric current', 'magnetic field', 'electric field',
        'chemical reaction', 'chemical bond', 'molecular structure',
        'periodic table', 'atomic number', 'atomic mass',
        'industrial revolution', 'scientific revolution', 'information age',
        'middle ages', 'roman empire', 'renaissance period',
    }
    
    # Stop words that shouldn't be concepts
    STOP_WORDS = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from', 'but', 'not',
        'are', 'was', 'were', 'has', 'had', 'have', 'can', 'may', 'will',
        'its', 'his', 'her', 'our', 'their', 'your', 'she', 'him', 'you',
        'who', 'how', 'why', 'what', 'when', 'where', 'which',
        'any', 'all', 'one', 'two', 'three', 'four', 'five',
        'yes', 'no', 'nor', 'yet', 'per', 'via', 'etc', 'viz',
        'about', 'also', 'into', 'onto', 'upon', 'through',
        'being', 'been', 'would', 'could', 'should', 'might',
        'there', 'these', 'those', 'here', 'very', 'much',
        'more', 'most', 'some', 'such', 'than', 'then',
        'only', 'just', 'even', 'also', 'well', 'back',
    }
    
    @classmethod
    def extract_concepts(cls, sentence: str) -> List[str]:
        """
        Extract high-quality concepts from sentence.
        
        Returns list of concepts, preferring multi-word compounds.
        """
        concepts = []
        sentence_lower = sentence.lower()
        
        # 1. Extract domain-specific compound terms
        for compound in cls.DOMAIN_COMPOUNDS:
            if compound in sentence_lower:
                # Get the actual casing from the sentence
                start_idx = sentence_lower.find(compound)
                end_idx = start_idx + len(compound)
                actual_text = sentence[start_idx:end_idx]
                concepts.append(actual_text.title())
        
        # 2. Extract multi-word proper nouns (capitalized sequences)
        # E.g., "William Shakespeare", "Mount Everest", "Amazon Rainforest"
        proper_noun_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        for match in re.finditer(proper_noun_pattern, sentence):
            concept = match.group(1)
            if concept not in concepts:
                concepts.append(concept)
        
        # 3. Extract quoted terms (often technical or specific concepts)
        quoted_pattern = r'"([^"]+)"'
        for match in re.finditer(quoted_pattern, sentence):
            concept = match.group(1).title()
            if concept not in concepts and len(concept) > 2:
                concepts.append(concept)
        
        # 4. Extract capitalized words (potential named entities)
        # But skip if they're part of a multi-word concept already extracted
        cap_pattern = r'\b([A-Z][a-z]{2,})\b'
        for match in re.finditer(cap_pattern, sentence):
            word = match.group(1)
            # Check if this word is already part of a multi-word concept
            if not any(word in concept for concept in concepts):
                concepts.append(word)
        
        # 5. Extract significant noun phrases (Adj + Noun patterns)
        # E.g., "genetic code", "cellular respiration", "binary system"
        adj_noun_pattern = r'\b([a-z]+(?:ic|al|ive|ous|ful|less|ary|ory))\s+([a-z]{4,})\b'
        for match in re.finditer(adj_noun_pattern, sentence_lower):
            adj = match.group(1)
            noun = match.group(2)
            if adj not in cls.STOP_WORDS and noun not in cls.STOP_WORDS:
                phrase = f"{adj} {noun}".title()
                if phrase not in concepts:
                    concepts.append(phrase)
        
        # 6. Extract technical terms with hyphens
        hyphen_pattern = r'\b([a-z]+-[a-z]+(?:-[a-z]+)*)\b'
        for match in re.finditer(hyphen_pattern, sentence_lower):
            term = match.group(1).title()
            if term not in concepts:
                concepts.append(term)
        
        # 7. Extract acronyms (2+ capital letters)
        acronym_pattern = r'\b([A-Z]{2,})\b'
        for match in re.finditer(acronym_pattern, sentence):
            acronym = match.group(1)
            if acronym not in concepts:
                concepts.append(acronym)
        
        # 8. Extract single significant nouns (length >= 5, not stop words)
        words = sentence.split()
        for word in words:
            # Clean up punctuation
            clean_word = word.strip('.,;:!?()[]{}\"\'').capitalize()
            
            # Skip if too short, stop word, or already captured
            if (len(clean_word) < 5 or 
                clean_word.lower() in cls.STOP_WORDS or 
                clean_word in concepts or
                any(clean_word.lower() in c.lower() for c in concepts)):
                continue
            
            # Add significant nouns
            concepts.append(clean_word)
        
        # 9. Clean and validate concepts
        cleaned_concepts = []
        for concept in concepts:
            # Remove concepts that are just stop words
            if concept.lower() in cls.STOP_WORDS:
                continue
            
            # Remove very short concepts (< 3 chars) unless they're acronyms
            if len(concept) < 3 and not concept.isupper():
                continue
            
            # Remove concepts that are too long (likely extraction errors)
            if len(concept) > 50:
                continue
            
            # Remove leading/trailing punctuation
            concept = concept.strip('.,;:!?()[]{}')
            
            if concept and concept not in cleaned_concepts:
                cleaned_concepts.append(concept)
        
        return cleaned_concepts
    
    @classmethod
    def extract_relations(
        cls,
        sentence: str,
        concepts: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Extract relations between concepts with better pattern matching.
        
        Returns list of (subject, relation, object) tuples.
        """
        relations = []
        sentence_lower = sentence.lower()
        
        # Build concept map (lowercase -> proper form)
        concept_map = {c.lower(): c for c in concepts}
        concept_lowers = list(concept_map.keys())
        
        # Strategy: For each pair of extracted concepts, check if they appear in sentence
        # with a relation word between them
        relation_keywords = {
            'is a': ['is a', 'is an', 'is the', 'are'],
            'causes': ['causes', 'cause', 'caused'],
            'leads_to': ['leads to', 'lead to', 'results in', 'result in'],
            'produces': ['produces', 'produce', 'generates', 'generate'],
            'converts_to': ['converts to', 'convert to', 'transforms into', 'transform into'],
            'made_of': ['made of', 'composed of', 'consists of'],
            'contains': ['contains', 'contain', 'includes', 'include'],
            'part_of': ['part of', 'component of'],
            'has_property': ['has', 'have', 'features', 'feature'],
            'enables': ['enables', 'enable', 'allows', 'allow', 'facilitates'],
            'located_in': ['located in', 'found in', 'occurs in'],
        }
        
        # Check each pair of concepts (order matters for subject/object)
        for i, subj_lower in enumerate(concept_lowers):
            subj = concept_map[subj_lower]
            
            # Skip if concept is too long (likely not a proper concept)
            if len(subj) > 30:
                continue
            
            for j, obj_lower in enumerate(concept_lowers):
                if i == j:  # Skip self-relations
                    continue
                
                obj = concept_map[obj_lower]
                
                # Skip if object is too long
                if len(obj) > 30:
                    continue
                
                # Find positions of concepts in sentence
                subj_pos = sentence_lower.find(subj_lower)
                obj_pos = sentence_lower.find(obj_lower)
                
                # Only look for relations if both concepts appear and subj comes first
                if subj_pos == -1 or obj_pos == -1 or subj_pos >= obj_pos:
                    continue
                
                # Extract text between the two concepts
                between_text = sentence_lower[subj_pos + len(subj_lower):obj_pos].strip()
                
                # Check if any relation keyword appears in the between text
                for relation_type, keywords in relation_keywords.items():
                    for keyword in keywords:
                        if keyword in between_text:
                            # Make sure it's a word boundary match (not substring)
                            # Simple check: surrounded by spaces or start/end
                            pattern = r'\b' + re.escape(keyword) + r'\b'
                            if re.search(pattern, between_text):
                                relations.append((subj, relation_type, obj))
                                break  # Only add one relation per concept pair
                    if relations and relations[-1][0] == subj and relations[-1][2] == obj:
                        break # Break outer loop too
        
        return relations


def patch_text_learner():
    """
    Patch the TextKnowledgeLearner to use enhanced concept extraction AND relation extraction.
    
    Call this before training to improve concept quality.
    """
    from python.core.language import text_knowledge_learner
    
    # Replace the _extract_concepts method
    original_extract_concepts = text_knowledge_learner.TextKnowledgeLearner._extract_concepts
    
    def enhanced_extract_concepts(self, sentence: str) -> List[str]:
        """Enhanced concept extraction using EnhancedConceptExtractor."""
        return EnhancedConceptExtractor.extract_concepts(sentence)
    
    text_knowledge_learner.TextKnowledgeLearner._extract_concepts = enhanced_extract_concepts
    
    # ALSO replace _extract_relations to preserve compound terms
    original_extract_relations = text_knowledge_learner.TextKnowledgeLearner._extract_relations
    
    def enhanced_extract_relations(self, sentence: str, concepts: List[str]) -> List[Tuple[str, str, str]]:
        """Enhanced relation extraction that preserves compound concepts."""
        # Use enhanced relation extractor
        relations = EnhancedConceptExtractor.extract_relations(sentence, concepts)
        
        # Filter out any relations with subjects/objects not in the concept list
        # This prevents single-word concepts from leaking in
        valid_relations = []
        for subj, rel, obj in relations:
            if subj in concepts and obj in concepts:
                valid_relations.append((subj, rel, obj))
        
        return valid_relations
    
    text_knowledge_learner.TextKnowledgeLearner._extract_relations = enhanced_extract_relations
    
    print("✓ TextKnowledgeLearner patched with enhanced concept AND relation extraction")
    return (original_extract_concepts, original_extract_relations)


if __name__ == '__main__':
    # Test the enhanced extractor
    test_sentences = [
        "Artificial intelligence is a simulation of human intelligence by machines.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "Climate change is caused by greenhouse gas emissions.",
        "DNA contains genetic instructions for all living organisms.",
        "The Solar System consists of the Sun and eight planets.",
    ]
    
    print("Enhanced Concept Extraction Test")
    print("=" * 80)
    
    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        concepts = EnhancedConceptExtractor.extract_concepts(sentence)
        print(f"Concepts: {concepts}")
        
        relations = EnhancedConceptExtractor.extract_relations(sentence, concepts)
        if relations:
            print("Relations:")
            for subj, rel, obj in relations:
                print(f"  - {subj} --[{rel}]--> {obj}")
