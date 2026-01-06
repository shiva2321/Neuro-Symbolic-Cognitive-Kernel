"""
Central Controller
Routes queries to appropriate specialized modules and manages the overall system.
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import glob

from modules.base_module import BaseModule
from modules.text_module import TextModule
from modules.math_module import MathModule
from modules.code_module import CodeModule


class CentralController:
    """
    Central controller that manages all specialized modules and routes queries.
    Implements Phase 2: Modularity and Integration
    """

    def __init__(self):
        """Initialize the central controller with all modules"""
        self.modules: List[BaseModule] = []
        self.module_map: Dict[str, BaseModule] = {}

        # Initialize specialized modules
        self._initialize_modules()

        # Training state
        self.is_trained = False
        self.training_data_path: Optional[str] = None

    def _initialize_modules(self):
        """Initialize all specialized modules"""
        print("Initializing specialized modules...")

        # Create modules
        text_module = TextModule("text_module")
        math_module = MathModule("math_module")
        code_module = CodeModule("code_module")

        # Register modules
        self.register_module(text_module)
        self.register_module(math_module)
        self.register_module(code_module)

        print(f"Initialized {len(self.modules)} modules")

    def register_module(self, module: BaseModule):
        """
        Register a new module with the controller.

        Args:
            module: Module to register
        """
        self.modules.append(module)
        self.module_map[module.name] = module
        print(f"  Registered: {module.name}")

    def route_query(self, query: str) -> Tuple[BaseModule, float]:
        """
        Route a query to the most appropriate module.

        Args:
            query: Input query

        Returns:
            Tuple of (selected_module, confidence_score)
        """
        # Get confidence scores from all modules
        scores = []
        for module in self.modules:
            confidence = module.can_handle(query)
            scores.append((module, confidence))

        # Sort by confidence
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return module with highest confidence
        return scores[0] if scores else (None, 0.0)

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query by routing it to the appropriate module.

        Args:
            query: Input query
            context: Optional context information

        Returns:
            Response dictionary from the selected module
        """
        # Route to appropriate module
        module, confidence = self.route_query(query)

        if module is None:
            return {
                'success': False,
                'error': 'No suitable module found',
                'response': None
            }

        print(f"Routing to {module.name} (confidence: {confidence:.2f})")

        # Process with selected module
        result = module.process(query, context)

        # Add routing information
        result['routing_info'] = {
            'selected_module': module.name,
            'confidence': confidence,
            'all_scores': {m.name: m.can_handle(query) for m in self.modules}
        }

        return result

    def load_training_data(self, data_path: str) -> Dict[str, int]:
        """
        Load training data from a directory containing books/documents.

        Args:
            data_path: Path to directory containing training data

        Returns:
            Statistics about loaded data
        """
        print(f"Loading training data from: {data_path}")

        self.training_data_path = data_path
        stats = {
            'text_files': 0,
            'code_files': 0,
            'math_files': 0,
            'total_files': 0
        }

        if not os.path.exists(data_path):
            print(f"Warning: Path does not exist: {data_path}")
            return stats

        # Load text files
        text_patterns = ['*.txt', '*.md', '*.doc','*.pdf']
        code_patterns = ['*.py', '*.js', '*.java', '*.cpp', '*.c', '*.cs']

        # Count files
        for pattern in text_patterns:
            files = glob.glob(os.path.join(data_path, '**', pattern), recursive=True)
            stats['text_files'] += len(files)

        for pattern in code_patterns:
            files = glob.glob(os.path.join(data_path, '**', pattern), recursive=True)
            stats['code_files'] += len(files)

        stats['total_files'] = stats['text_files'] + stats['code_files']

        print(f"Found {stats['total_files']} files:")
        print(f"  Text files: {stats['text_files']}")
        print(f"  Code files: {stats['code_files']}")

        return stats

    def train(self, data_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Train all modules on the provided dataset.
        Implements Phase 4: Training and Updating

        Args:
            data_path: Optional path to training data (uses loaded path if None)
            **kwargs: Additional training parameters

        Returns:
            Training statistics for all modules
        """
        if data_path:
            self.load_training_data(data_path)

        if not self.training_data_path:
            return {
                'success': False,
                'error': 'No training data loaded. Call load_training_data() first.'
            }

        print("\n" + "="*60)
        print("STARTING MODULAR NETWORK TRAINING")
        print("="*60 + "\n")

        training_results = {}

        # Load and categorize files
        text_files, code_files, math_files = self._categorize_training_files()

        # Train Text Module
        if text_files:
            print("\n--- Training Text Module ---")
            text_data = self._load_files(text_files[:1000])  # Limit to 1000 files
            text_stats = self.module_map['text_module'].train(
                text_data,
                epochs=kwargs.get('epochs', 5),
                learning_rate=kwargs.get('learning_rate', 0.1)
            )
            training_results['text_module'] = text_stats

        # Train Code Module
        if code_files:
            print("\n--- Training Code Module ---")
            code_data = self._load_files(code_files[:500])  # Limit to 500 files
            code_stats = self.module_map['code_module'].train(code_data)
            training_results['code_module'] = code_stats

        # Train Math Module
        # For math, we can use text files that contain mathematical content
        if math_files or text_files:
            print("\n--- Training Math Module ---")
            math_data = self._load_files(math_files[:200]) if math_files else text_data[:100]
            math_stats = self.module_map['math_module'].train(math_data)
            training_results['math_module'] = math_stats

        self.is_trained = True

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60 + "\n")

        return {
            'success': True,
            'modules_trained': len(training_results),
            'results': training_results,
            'summary': self.get_system_statistics()
        }

    def _categorize_training_files(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Categorize training files by type.

        Returns:
            Tuple of (text_files, code_files, math_files)
        """
        text_files = []
        code_files = []
        math_files = []

        # Text patterns
        for pattern in ['*.txt', '*.md']:
            files = glob.glob(
                os.path.join(self.training_data_path, '**', pattern),
                recursive=True
            )
            text_files.extend(files)

        # Code patterns
        for pattern in ['*.py', '*.js', '*.java', '*.cpp', '*.c']:
            files = glob.glob(
                os.path.join(self.training_data_path, '**', pattern),
                recursive=True
            )
            code_files.extend(files)

        # Math files (look for specific keywords in text files)
        for filepath in text_files[:]:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1000)  # Read first 1000 chars
                    if any(word in content.lower() for word in ['theorem', 'equation', 'calculus', 'algebra', 'mathematics']):
                        math_files.append(filepath)
            except:
                pass

        return text_files, code_files, math_files

    def _load_files(self, filepaths: List[str]) -> List[str]:
        """
        Load content from multiple files.

        Args:
            filepaths: List of file paths

        Returns:
            List of file contents
        """
        contents = []

        for filepath in filepaths:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        contents.append(content)
            except Exception as e:
                print(f"  Warning: Could not read {filepath}: {e}")

        return contents

    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the entire system.
        Implements Phase 3: Scalability and Performance monitoring
        """
        stats = {
            'is_trained': self.is_trained,
            'num_modules': len(self.modules),
            'modules': {}
        }

        for module in self.modules:
            stats['modules'][module.name] = module.get_statistics()

        return stats

    def save_system(self, save_dir: str):
        """
        Save all modules to disk.

        Args:
            save_dir: Directory to save modules
        """
        os.makedirs(save_dir, exist_ok=True)

        for module in self.modules:
            filepath = os.path.join(save_dir, f"{module.name}.pkl")
            module.save(filepath)
            print(f"Saved {module.name} to {filepath}")

    def load_system(self, save_dir: str):
        """
        Load all modules from disk.

        Args:
            save_dir: Directory containing saved modules
        """
        for module in self.modules:
            filepath = os.path.join(save_dir, f"{module.name}.pkl")
            if os.path.exists(filepath):
                module.load(filepath)
                print(f"Loaded {module.name} from {filepath}")
            else:
                print(f"Warning: Could not find saved module: {filepath}")

        self.is_trained = any(m.is_trained for m in self.modules)

    def interactive_mode(self):
        """
        Run interactive mode for testing the system.
        """
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("Enter queries to test the system.")
        print("Commands: 'quit' to exit, 'stats' for statistics")
        print("="*60 + "\n")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if not query:
                    continue

                if query.lower() == 'quit':
                    print("Exiting interactive mode...")
                    break

                if query.lower() == 'stats':
                    stats = self.get_system_statistics()
                    print("\nSystem Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                # Process query
                result = self.process(query)

                print(f"\n{'='*60}")
                print(f"Module: {result.get('module', 'unknown')}")
                print(f"Success: {result.get('success', False)}")
                print(f"{'='*60}")

                if result.get('success'):
                    print(f"\nResponse:\n{result.get('response', 'No response')}")
                else:
                    print(f"\nError: {result.get('error', 'Unknown error')}")

                if 'routing_info' in result:
                    print(f"\nRouting Info:")
                    for key, value in result['routing_info'].items():
                        print(f"  {key}: {value}")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")

    def __repr__(self):
        return f"CentralController(modules={len(self.modules)}, trained={self.is_trained})"

