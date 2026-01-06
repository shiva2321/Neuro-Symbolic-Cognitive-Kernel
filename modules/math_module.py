"""
Math Module
Specialized module for mathematical computations and equation processing.
"""

import re
from typing import Dict, List, Optional, Any
import numpy as np

from modules.base_module import BaseModule, ModuleType
from core.graph_network import DomainType, Node, Edge, NodeType
from utils.text_processor import TextProcessor


class MathModule(BaseModule):
    """
    Specialized module for mathematical computations and processing.
    """

    def __init__(self, name: str = "math_module"):
        super().__init__(ModuleType.MATH, name)
        self.text_processor = TextProcessor(use_spacy=False, min_word_frequency=1)

        # Math-specific patterns
        self.math_patterns = [
            r'\b(calculate|compute|solve|evaluate|find)\b',
            r'\b(equation|derivative|integral|sum|product)\b',
            r'\b(theorem|proof|formula)\b',
            r'[+\-*/=∫∑√²³]',
            r'\d+\s*[+\-*/]\s*\d+',
        ]

        # Common math operations
        self.operations = {
            'add': lambda a, b: a + b,
            'subtract': lambda a, b: a - b,
            'multiply': lambda a, b: a * b,
            'divide': lambda a, b: a / b if b != 0 else float('inf'),
            'power': lambda a, b: a ** b,
        }

    def can_handle(self, query: str) -> float:
        """
        Determine if this is a math query.

        Returns confidence score based on math-related patterns.
        """
        query_lower = query.lower()

        confidence = 0.0

        # Check for math patterns
        for pattern in self.math_patterns:
            if re.search(pattern, query_lower):
                confidence += 0.25

        # Check for numbers
        if re.search(r'\d+', query_lower):
            confidence += 0.2

        # Check for math operations
        math_ops = ['plus', 'minus', 'times', 'divided', 'equals', 'squared', 'cubed']
        for op in math_ops:
            if op in query_lower:
                confidence += 0.15

        return min(confidence, 1.0)

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a math query.

        Args:
            query: Input query
            context: Optional context

        Returns:
            Dictionary with computation result and metadata
        """
        try:
            # Try to parse and evaluate simple arithmetic
            result = self._evaluate_expression(query)

            if result is not None:
                return {
                    'success': True,
                    'response': f"The result is: {result}",
                    'module': self.name,
                    'computation': result,
                    'metadata': {
                        'query': query,
                        'type': 'arithmetic'
                    }
                }

            # Try symbolic math
            symbolic_result = self._process_symbolic(query)

            if symbolic_result:
                return {
                    'success': True,
                    'response': symbolic_result,
                    'module': self.name,
                    'metadata': {
                        'query': query,
                        'type': 'symbolic'
                    }
                }

            # Fallback to graph-based generation
            if self.is_trained:
                response = self._generate_math_explanation(query)
                return {
                    'success': True,
                    'response': response,
                    'module': self.name,
                    'metadata': {
                        'query': query,
                        'type': 'explanation'
                    }
                }

            return {
                'success': False,
                'error': 'Could not process math query',
                'response': None
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': None
            }

    def train(self, training_data: List[str], **kwargs) -> Dict[str, Any]:
        """
        Train the math module on mathematical texts.

        Args:
            training_data: List of math texts (textbooks, problems, solutions)
            **kwargs: Additional parameters

        Returns:
            Training statistics
        """
        print(f"Training {self.name} on {len(training_data)} documents...")

        total_stats = {'nodes_added': 0, 'edges_added': 0, 'tokens_processed': 0}

        for i, text in enumerate(training_data):
            if i % 50 == 0:
                print(f"  Processing document {i+1}/{len(training_data)}")

            stats = self.text_processor.process_text(
                text,
                self.graph,
                domain=DomainType.MATH
            )

            for key in total_stats:
                total_stats[key] += stats[key]

        # Add math-specific nodes
        self._add_math_operators()

        self.is_trained = True

        return {
            'processing_stats': total_stats,
            'final_graph_stats': self.graph.get_statistics()
        }

    def _evaluate_expression(self, query: str) -> Optional[float]:
        """
        Evaluate simple arithmetic expressions.
        """
        # Extract numbers and operators
        # Simple patterns like "2 + 3", "10 * 5", etc.

        patterns = [
            (r'(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)', lambda m: float(m.group(1)) + float(m.group(2))),
            (r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', lambda m: float(m.group(1)) - float(m.group(2))),
            (r'(\d+\.?\d*)\s*\*\s*(\d+\.?\d*)', lambda m: float(m.group(1)) * float(m.group(2))),
            (r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)', lambda m: float(m.group(1)) / float(m.group(2)) if float(m.group(2)) != 0 else float('inf')),
            (r'(\d+\.?\d*)\s*\^\s*(\d+\.?\d*)', lambda m: float(m.group(1)) ** float(m.group(2))),
        ]

        for pattern, evaluator in patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    return evaluator(match)
                except:
                    pass

        # Try eval for simple expressions (with safety checks)
        try:
            # Extract mathematical expression
            expr = re.search(r'[\d+\-*/().\s]+', query)
            if expr:
                expr_str = expr.group(0).strip()
                # Safety check - only allow numbers and operators
                if re.match(r'^[\d+\-*/().\s]+$', expr_str):
                    result = eval(expr_str)
                    return float(result)
        except:
            pass

        return None

    def _process_symbolic(self, query: str) -> Optional[str]:
        """
        Process symbolic math queries (derivatives, integrals, etc.)
        """
        query_lower = query.lower()

        # Derivative patterns
        if 'derivative' in query_lower:
            # Extract function
            if 'x^2' in query_lower or 'x²' in query_lower:
                return "The derivative of x² is 2x"
            elif 'x^3' in query_lower or 'x³' in query_lower:
                return "The derivative of x³ is 3x²"
            elif re.search(r'x\^(\d+)', query_lower):
                match = re.search(r'x\^(\d+)', query_lower)
                n = int(match.group(1))
                return f"The derivative of x^{n} is {n}x^{n-1}"

        # Integral patterns
        if 'integral' in query_lower:
            if 'x' in query_lower and 'dx' in query_lower:
                return "The integral of x dx is (x²/2) + C"

        return None

    def _generate_math_explanation(self, query: str) -> str:
        """
        Generate mathematical explanation using the graph.
        """
        # Find relevant math nodes
        query_tokens = self.text_processor.tokenize(query)

        relevant_nodes = []
        for token in query_tokens:
            for node_id, node in self.graph.nodes.items():
                if token.lower() in node.value.lower():
                    relevant_nodes.append(node)

        if relevant_nodes:
            # Use highest frequency node
            start_node = max(relevant_nodes, key=lambda n: n.frequency)

            # Generate explanation (simplified)
            return f"Mathematical concept related to: {start_node.value}"

        return "Mathematical query processed"

    def _add_math_operators(self):
        """Add common mathematical operators and symbols as nodes"""
        operators = [
            ('+', 'addition', NodeType.SPECIAL),
            ('-', 'subtraction', NodeType.SPECIAL),
            ('*', 'multiplication', NodeType.SPECIAL),
            ('/', 'division', NodeType.SPECIAL),
            ('=', 'equals', NodeType.SPECIAL),
            ('^', 'power', NodeType.SPECIAL),
            ('√', 'square_root', NodeType.SPECIAL),
            ('∫', 'integral', NodeType.SPECIAL),
            ('∑', 'sum', NodeType.SPECIAL),
        ]

        for symbol, name, node_type in operators:
            node = Node(
                id=f"math_op_{name}",
                value=symbol,
                node_type=node_type,
                domain=DomainType.MATH,
                frequency=100
            )
            self.graph.add_node(node)

