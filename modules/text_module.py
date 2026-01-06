"""
Text Generation Module
Specialized module for natural language text generation and processing.
"""

import re
from typing import Dict, List, Optional, Any

from modules.base_module import BaseModule, ModuleType
from core.traversal_engine import TraversalEngine, TraversalConfig, TraversalStrategy
from core.training_engine import TrainingEngine, TrainingConfig
from utils.text_processor import TextProcessor
from core.graph_network import DomainType


class TextModule(BaseModule):
    """
    Specialized module for text generation and natural language processing.
    """

    def __init__(self, name: str = "text_module"):
        super().__init__(ModuleType.TEXT, name)
        self.text_processor = TextProcessor(use_spacy=False, min_word_frequency=2)
        self.traversal_engine = None
        self.training_engine = None

        # Text-specific patterns
        self.text_patterns = [
            r'\b(write|generate|create|compose|tell|describe)\b',
            r'\b(story|paragraph|essay|article|text|narrative)\b',
            r'\b(about|regarding|concerning)\b',
        ]

    def can_handle(self, query: str) -> float:
        """
        Determine if this is a text generation query.

        Returns confidence score based on text-related keywords.
        """
        query_lower = query.lower()

        # Check for text generation patterns
        confidence = 0.0

        for pattern in self.text_patterns:
            if re.search(pattern, query_lower):
                confidence += 0.3

        # Check for absence of code/math indicators
        code_indicators = ['function', 'class', 'def ', 'code', 'program', '==', '!=']
        math_indicators = ['calculate', 'solve', 'equation', 'derivative', '∫', '=']

        has_code = any(indicator in query_lower for indicator in code_indicators)
        has_math = any(indicator in query_lower for indicator in math_indicators)

        if not has_code and not has_math:
            confidence += 0.3

        # Generic text queries default to text module
        if confidence == 0 and len(query.split()) > 3:
            confidence = 0.4

        return min(confidence, 1.0)

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a text generation query.

        Args:
            query: Input query
            context: Optional context (previous messages, etc.)

        Returns:
            Dictionary with generated text and metadata
        """
        if not self.is_trained:
            return {
                'success': False,
                'error': 'Module not trained yet',
                'response': None
            }

        # Initialize traversal engine if needed
        if self.traversal_engine is None:
            self.traversal_engine = TraversalEngine(
                self.graph,
                TraversalConfig(
                    strategy=TraversalStrategy.TEMPERATURE,
                    max_tokens=100,
                    temperature=0.8,
                    repetition_penalty=1.3
                )
            )

        try:
            # Extract starting token from query
            start_token = self._extract_start_token(query, context)

            # Generate text
            generated_text = self.traversal_engine.generate_text(start_token)

            return {
                'success': True,
                'response': generated_text,
                'module': self.name,
                'start_token': start_token,
                'metadata': {
                    'tokens_generated': len(generated_text.split()),
                    'strategy': 'temperature_sampling'
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': None
            }

    def train(self, training_data: List[str], **kwargs) -> Dict[str, Any]:
        """
        Train the text module on text data.

        Args:
            training_data: List of text documents
            **kwargs: Additional parameters (epochs, learning_rate, etc.)

        Returns:
            Training statistics
        """
        print(f"Training {self.name} on {len(training_data)} documents...")

        # Process all training data
        total_stats = {'nodes_added': 0, 'edges_added': 0, 'tokens_processed': 0}

        for i, text in enumerate(training_data):
            if i % 100 == 0:
                print(f"  Processing document {i+1}/{len(training_data)}")

            stats = self.text_processor.process_text(
                text,
                self.graph,
                domain=DomainType.TEXT
            )

            for key in total_stats:
                total_stats[key] += stats[key]

        print(f"Graph construction complete: {self.graph}")

        # Extract sequences for training
        sequences = self._extract_training_sequences(training_data)

        # Initialize training engine
        training_config = TrainingConfig(
            learning_rate=kwargs.get('learning_rate', 0.1),
            num_epochs=kwargs.get('epochs', 5),
            batch_size=kwargs.get('batch_size', 32)
        )

        self.training_engine = TrainingEngine(self.graph, training_config)

        # Train
        metrics = self.training_engine.train_from_sequences(sequences)

        # Update context vectors
        self.training_engine.update_context_vectors()

        self.is_trained = True

        return {
            'processing_stats': total_stats,
            'training_metrics': [str(m) for m in metrics],
            'final_graph_stats': self.graph.get_statistics(),
            'training_summary': self.training_engine.get_training_summary()
        }

    def _extract_start_token(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """
        Extract or infer a good starting token from the query.
        """
        # Remove common query prefixes
        query = re.sub(r'^(write|generate|create|tell me|describe)\s+', '', query.lower())
        query = re.sub(r'^(a|an|the)\s+', '', query)

        # Tokenize
        tokens = self.text_processor.tokenize(query)

        # Find a token that exists in the graph
        for token in tokens:
            for node_id, node in self.graph.nodes.items():
                if node.value.lower() == token.lower():
                    return token

        # Fallback to common starting words
        common_starts = ['the', 'a', 'in', 'on', 'once', 'there', 'it', 'this']
        for start in common_starts:
            for node_id, node in self.graph.nodes.items():
                if node.value.lower() == start:
                    return start

        # Last resort: use first node in graph
        if self.graph.nodes:
            first_node = list(self.graph.nodes.values())[0]
            return first_node.value

        return "the"

    def _extract_training_sequences(self, texts: List[str], max_sequences: int = 10000) -> List[List[str]]:
        """
        Extract token sequences from texts for training.
        """
        sequences = []

        for text in texts:
            tokens = self.text_processor.tokenize(text)

            # Create overlapping sequences of length 10-20
            for seq_len in [10, 15, 20]:
                for i in range(0, len(tokens) - seq_len + 1, 5):
                    sequence = tokens[i:i + seq_len]
                    sequences.append(sequence)

                    if len(sequences) >= max_sequences:
                        return sequences

        return sequences

    def generate_multiple_variants(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate multiple different text variants for the same query.
        """
        if not self.traversal_engine:
            return []

        start_token = self._extract_start_token(query, None)

        variants = self.traversal_engine.get_multiple_completions(
            start_token,
            num_completions=num_variants,
            config=TraversalConfig(
                strategy=TraversalStrategy.TEMPERATURE,
                temperature=1.0,
                max_tokens=100
            )
        )

        return variants

