"""
Code Module
Specialized module for code generation and programming assistance.
"""

import re
from typing import Dict, List, Optional, Any

from modules.base_module import BaseModule, ModuleType
from core.graph_network import DomainType, Node, NodeType
from utils.text_processor import TextProcessor
from core.traversal_engine import TraversalEngine, TraversalConfig, TraversalStrategy


class CodeModule(BaseModule):
    """
    Specialized module for code generation and programming assistance.
    """

    def __init__(self, name: str = "code_module"):
        super().__init__(ModuleType.CODE, name)
        self.text_processor = TextProcessor(use_spacy=False, min_word_frequency=1)
        self.traversal_engine = None

        # Code-specific patterns
        self.code_patterns = [
            r'\b(write|create|implement|code|program)\b.*\b(function|class|method|script)\b',
            r'\b(python|javascript|java|c\+\+|code)\b',
            r'\b(algorithm|implement|debug|fix)\b',
            r'(def |function |class |import |return |if |else |for |while )',
        ]

        # Programming language keywords
        self.language_keywords = {
            'python': ['def', 'class', 'import', 'return', 'if', 'else', 'for', 'while', 'try', 'except'],
            'javascript': ['function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while'],
            'java': ['public', 'private', 'class', 'void', 'return', 'if', 'else', 'for', 'while'],
        }

    def can_handle(self, query: str) -> float:
        """
        Determine if this is a coding query.

        Returns confidence score based on code-related patterns.
        """
        query_lower = query.lower()

        confidence = 0.0

        # Check for code patterns
        for pattern in self.code_patterns:
            if re.search(pattern, query_lower):
                confidence += 0.3

        # Check for programming language mentions
        languages = ['python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'go', 'rust']
        for lang in languages:
            if lang in query_lower:
                confidence += 0.25

        # Check for code-specific words
        code_words = ['function', 'algorithm', 'implementation', 'syntax', 'debug', 'compile']
        for word in code_words:
            if word in query_lower:
                confidence += 0.15

        return min(confidence, 1.0)

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a code generation query.

        Args:
            query: Input query
            context: Optional context

        Returns:
            Dictionary with generated code and metadata
        """
        if not self.is_trained:
            # Return template code even if not trained
            return self._generate_template_code(query)

        try:
            # Detect programming language
            language = self._detect_language(query)

            # Initialize traversal engine if needed
            if self.traversal_engine is None:
                self.traversal_engine = TraversalEngine(
                    self.graph,
                    TraversalConfig(
                        strategy=TraversalStrategy.GREEDY,
                        max_tokens=50,
                        temperature=0.5
                    )
                )

            # Extract function/class name
            entity_name = self._extract_entity_name(query)

            # Generate code
            generated_code = self._generate_code(query, language, entity_name)

            return {
                'success': True,
                'response': generated_code,
                'module': self.name,
                'metadata': {
                    'language': language,
                    'entity_name': entity_name,
                    'type': 'code_generation'
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
        Train the code module on code samples.

        Args:
            training_data: List of code samples
            **kwargs: Additional parameters

        Returns:
            Training statistics
        """
        print(f"Training {self.name} on {len(training_data)} code samples...")

        total_stats = {'nodes_added': 0, 'edges_added': 0, 'tokens_processed': 0}

        for i, code in enumerate(training_data):
            if i % 50 == 0:
                print(f"  Processing code sample {i+1}/{len(training_data)}")

            # Process code preserving structure
            stats = self._process_code_sample(code)

            for key in total_stats:
                total_stats[key] += stats[key]

        # Add programming keywords
        self._add_code_keywords()

        self.is_trained = True

        return {
            'processing_stats': total_stats,
            'final_graph_stats': self.graph.get_statistics()
        }

    def _detect_language(self, query: str) -> str:
        """Detect programming language from query"""
        query_lower = query.lower()

        language_indicators = {
            'python': ['python', 'py', 'def ', 'import '],
            'javascript': ['javascript', 'js', 'function', 'const ', 'let '],
            'java': ['java', 'public class', 'public static'],
            'cpp': ['c++', 'cpp', 'std::', '#include'],
        }

        for lang, indicators in language_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    return lang

        return 'python'  # Default

    def _extract_entity_name(self, query: str) -> str:
        """Extract function or class name from query"""
        # Look for patterns like "write a function called X" or "create a class X"
        patterns = [
            r'function\s+(?:called\s+)?(\w+)',
            r'class\s+(?:called\s+)?(\w+)',
            r'method\s+(?:called\s+)?(\w+)',
            r'(?:called|named)\s+(\w+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)

        # Extract verb for default name
        verbs = re.findall(r'\b(sort|search|find|calculate|compute|process|handle)\w*\b', query, re.IGNORECASE)
        if verbs:
            return verbs[0].lower()

        return 'process'

    def _generate_code(self, query: str, language: str, entity_name: str) -> str:
        """Generate code based on query"""
        query_lower = query.lower()

        # Determine code type
        if 'function' in query_lower or 'method' in query_lower:
            return self._generate_function(language, entity_name, query)
        elif 'class' in query_lower:
            return self._generate_class(language, entity_name, query)
        else:
            return self._generate_function(language, entity_name, query)

    def _generate_function(self, language: str, name: str, query: str) -> str:
        """Generate a function based on language"""
        templates = {
            'python': f"""def {name}(data):
    \"\"\"
    {query}
    
    Args:
        data: Input data
    
    Returns:
        Processed result
    \"\"\"
    # TODO: Implement function logic
    result = None
    return result
""",
            'javascript': f"""function {name}(data) {{
    /**
     * {query}
     * 
     * @param {{any}} data - Input data
     * @returns {{any}} Processed result
     */
    // TODO: Implement function logic
    let result = null;
    return result;
}}
""",
            'java': f"""public static Object {name}(Object data) {{
    /**
     * {query}
     * 
     * @param data Input data
     * @return Processed result
     */
    // TODO: Implement method logic
    Object result = null;
    return result;
}}
"""
        }

        return templates.get(language, templates['python'])

    def _generate_class(self, language: str, name: str, query: str) -> str:
        """Generate a class based on language"""
        # Capitalize class name
        class_name = name.capitalize()

        templates = {
            'python': f"""class {class_name}:
    \"\"\"
    {query}
    \"\"\"
    
    def __init__(self):
        \"\"\"Initialize the {class_name}\"\"\"
        pass
    
    def process(self, data):
        \"\"\"Process data\"\"\"
        # TODO: Implement processing logic
        return None
""",
            'javascript': f"""class {class_name} {{
    /**
     * {query}
     */
    
    constructor() {{
        // Initialize properties
    }}
    
    process(data) {{
        // TODO: Implement processing logic
        return null;
    }}
}}
""",
            'java': f"""public class {class_name} {{
    /**
     * {query}
     */
    
    public {class_name}() {{
        // Constructor
    }}
    
    public Object process(Object data) {{
        // TODO: Implement processing logic
        return null;
    }}
}}
"""
        }

        return templates.get(language, templates['python'])

    def _generate_template_code(self, query: str) -> Dict[str, Any]:
        """Generate template code when module is not trained"""
        language = self._detect_language(query)
        entity_name = self._extract_entity_name(query)
        code = self._generate_code(query, language, entity_name)

        return {
            'success': True,
            'response': code,
            'module': self.name,
            'metadata': {
                'language': language,
                'entity_name': entity_name,
                'type': 'template',
                'note': 'Generated from template (module not trained)'
            }
        }

    def _process_code_sample(self, code: str) -> Dict[str, int]:
        """Process a code sample and add to graph"""
        stats = self.text_processor.process_text(
            code,
            self.graph,
            domain=DomainType.CODE
        )

        return stats

    def _add_code_keywords(self):
        """Add programming keywords as nodes"""
        all_keywords = set()
        for keywords in self.language_keywords.values():
            all_keywords.update(keywords)

        for keyword in all_keywords:
            node = Node(
                id=f"code_keyword_{keyword}",
                value=keyword,
                node_type=NodeType.SPECIAL,
                domain=DomainType.CODE,
                frequency=100
            )
            self.graph.add_node(node)

