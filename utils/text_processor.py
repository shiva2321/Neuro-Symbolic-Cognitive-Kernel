"""
Text Processor
Handles text preprocessing, tokenization, and graph construction from text data.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import string

from core.graph_network import GraphNetwork, Node, Edge, NodeType, DomainType


class TextProcessor:
    """
    Processes text data and constructs graph representations.
    """

    def __init__(self,
                 use_spacy: bool = False,
                 ngram_sizes: List[int] = None,
                 min_word_frequency: int = 2):
        """
        Initialize text processor.

        Args:
            use_spacy: Whether to use spaCy for advanced NLP (slower but better)
            ngram_sizes: List of n-gram sizes to extract [1, 2, 3]
            min_word_frequency: Minimum frequency for a word to be included
        """
        self.use_spacy = use_spacy
        self.ngram_sizes = ngram_sizes or [1, 2, 3]
        self.min_word_frequency = min_word_frequency

        self.word_counts = Counter()
        self.ngram_counts = defaultdict(Counter)
        self.co_occurrence = defaultdict(Counter)

        # Try to load spaCy if requested
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Warning: spaCy not available, falling back to basic tokenization")
                self.use_spacy = False

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words/tokens.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
        else:
            # Basic tokenization
            # Preserve punctuation as separate tokens
            text = re.sub(r'([.!?,;:()])', r' \1 ', text)
            tokens = text.split()

        return [token for token in tokens if token.strip()]

    def extract_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from token list"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def detect_domain(self, text: str) -> DomainType:
        """
        Detect the domain of the text (text, math, code).

        Args:
            text: Input text

        Returns:
            Detected domain type
        """
        text_lower = text.lower()

        # Math indicators
        math_indicators = ['+', '-', '*', '/', '=', '∫', '∑', '√', '²', '³',
                          'equation', 'theorem', 'derivative', 'integral']
        math_score = sum(1 for indicator in math_indicators if indicator in text_lower)

        # Code indicators
        code_indicators = ['def ', 'class ', 'import ', 'function', 'return ',
                          'if ', 'else:', 'for ', 'while ', '==', '!=', '{', '}',
                          'print(', 'console.log']
        code_score = sum(1 for indicator in code_indicators if indicator in text_lower)

        # Determine domain
        if code_score > math_score and code_score >= 2:
            return DomainType.CODE
        elif math_score > code_score and math_score >= 2:
            return DomainType.MATH
        elif code_score > 0 and math_score > 0:
            return DomainType.MIXED
        else:
            return DomainType.TEXT

    def process_text(self, text: str, graph: GraphNetwork,
                    domain: Optional[DomainType] = None) -> Dict[str, int]:
        """
        Process text and add nodes/edges to the graph.

        Args:
            text: Input text to process
            graph: Graph to add nodes/edges to
            domain: Optional domain override

        Returns:
            Statistics about processing
        """
        stats = {'nodes_added': 0, 'edges_added': 0, 'tokens_processed': 0}

        # Detect domain if not provided
        if domain is None:
            domain = self.detect_domain(text)

        # Tokenize
        tokens = self.tokenize(text)
        stats['tokens_processed'] = len(tokens)

        # Process unigrams (individual tokens)
        for token in tokens:
            self.word_counts[token] += 1

        # Create nodes for frequent words
        for token in tokens:
            if self.word_counts[token] >= self.min_word_frequency:
                node_id = self._create_node_id(token)

                node = Node(
                    id=node_id,
                    value=token,
                    node_type=self._get_node_type(token),
                    domain=domain,
                    frequency=self.word_counts[token]
                )

                if graph.add_node(node):
                    stats['nodes_added'] += 1

        # Create edges based on sequential relationships
        for i in range(len(tokens) - 1):
            current_token = tokens[i]
            next_token = tokens[i + 1]

            if (self.word_counts[current_token] >= self.min_word_frequency and
                self.word_counts[next_token] >= self.min_word_frequency):

                current_id = self._create_node_id(current_token)
                next_id = self._create_node_id(next_token)

                # Track co-occurrence
                self.co_occurrence[current_id][next_id] += 1

                edge = Edge(
                    source=current_id,
                    target=next_id,
                    edge_type="sequential",
                    co_occurrence=self.co_occurrence[current_id][next_id]
                )

                if graph.add_edge(edge):
                    stats['edges_added'] += 1

        # Process n-grams for phrase detection
        for n in self.ngram_sizes:
            if n > 1:
                ngrams = self.extract_ngrams(tokens, n)
                for ngram in ngrams:
                    self.ngram_counts[n][ngram] += 1

                    # Create phrase nodes for common n-grams
                    if self.ngram_counts[n][ngram] >= self.min_word_frequency * 2:
                        phrase = ' '.join(ngram)
                        phrase_id = self._create_node_id(phrase)

                        phrase_node = Node(
                            id=phrase_id,
                            value=phrase,
                            node_type=NodeType.PHRASE,
                            domain=domain,
                            frequency=self.ngram_counts[n][ngram]
                        )

                        if graph.add_node(phrase_node):
                            stats['nodes_added'] += 1

        return stats

    def process_book(self, book_text: str, graph: GraphNetwork,
                    chunk_size: int = 10000) -> Dict[str, int]:
        """
        Process a book by splitting it into chunks.

        Args:
            book_text: Full text of the book
            graph: Graph to add nodes/edges to
            chunk_size: Size of text chunks to process at once

        Returns:
            Aggregate statistics
        """
        total_stats = {'nodes_added': 0, 'edges_added': 0, 'tokens_processed': 0}

        # Split into chunks
        chunks = [book_text[i:i+chunk_size] for i in range(0, len(book_text), chunk_size)]

        for chunk in chunks:
            stats = self.process_text(chunk, graph)
            for key in total_stats:
                total_stats[key] += stats[key]

        return total_stats

    def _create_node_id(self, token: str) -> str:
        """Create a unique node ID from a token"""
        # Normalize the token
        normalized = token.lower().strip()
        # Create ID (could use hash for very large graphs)
        return f"node_{normalized}_{hash(normalized) % 1000000}"

    def _get_node_type(self, token: str) -> NodeType:
        """Determine the type of node based on the token"""
        if token in string.punctuation:
            return NodeType.PUNCTUATION
        elif token.startswith('<') and token.endswith('>'):
            return NodeType.SPECIAL
        elif ' ' in token:
            return NodeType.PHRASE
        elif len(token.split()) > 1:
            return NodeType.PHRASE
        else:
            return NodeType.WORD

    def add_semantic_edges(self, graph: GraphNetwork, similarity_threshold: float = 0.7):
        """
        Add semantic edges between similar nodes based on context vectors.
        This is called after initial graph construction.

        Args:
            graph: Graph to add semantic edges to
            similarity_threshold: Minimum cosine similarity for edge creation
        """
        nodes = list(graph.nodes.values())
        edges_added = 0

        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Skip if already connected
                if graph.get_edge(node1.id, node2.id):
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(
                    node1.context_vector,
                    node2.context_vector
                )

                if similarity >= similarity_threshold:
                    edge = Edge(
                        source=node1.id,
                        target=node2.id,
                        weight=similarity,
                        edge_type="semantic"
                    )

                    if graph.add_edge(edge):
                        edges_added += 1

        return edges_added

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if vec1 is None or vec2 is None:
            return 0.0

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def get_statistics(self) -> Dict[str, any]:
        """Get processing statistics"""
        return {
            'total_words': len(self.word_counts),
            'total_unique_words': len(self.word_counts),
            'most_common_words': self.word_counts.most_common(10),
            'ngram_stats': {
                n: len(self.ngram_counts[n])
                for n in self.ngram_counts
            }
        }

