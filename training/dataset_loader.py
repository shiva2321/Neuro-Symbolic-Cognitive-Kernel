"""
Dataset Loader
Handles loading and managing training datasets.
Implements Phase 4: Training data management
"""

import os
import glob
from typing import List, Dict, Optional, Tuple
import random


class DatasetLoader:
    """
    Loads and manages training datasets for the neural network.
    """

    def __init__(self, data_dir: str):
        """
        Initialize dataset loader.

        Args:
            data_dir: Directory containing training data
        """
        self.data_dir = data_dir
        self.text_files: List[str] = []
        self.code_files: List[str] = []
        self.loaded_data: Dict[str, List[str]] = {
            'text': [],
            'code': [],
            'math': []
        }

    def scan_directory(self) -> Dict[str, int]:
        """
        Scan directory for training files.

        Returns:
            Statistics about found files
        """
        print(f"Scanning directory: {self.data_dir}")

        if not os.path.exists(self.data_dir):
            print(f"Warning: Directory does not exist: {self.data_dir}")
            return {'total': 0}

        # Text file patterns
        text_patterns = ['*.txt', '*.md', '*.text', '*.doc']
        code_patterns = ['*.py', '*.js', '*.java', '*.cpp', '*.c', '*.cs', '*.rb', '*.go']

        # Find files
        for pattern in text_patterns:
            files = glob.glob(os.path.join(self.data_dir, '**', pattern), recursive=True)
            self.text_files.extend(files)

        for pattern in code_patterns:
            files = glob.glob(os.path.join(self.data_dir, '**', pattern), recursive=True)
            self.code_files.extend(files)

        stats = {
            'text_files': len(self.text_files),
            'code_files': len(self.code_files),
            'total': len(self.text_files) + len(self.code_files)
        }

        print(f"Found {stats['total']} files:")
        print(f"  Text files: {stats['text_files']}")
        print(f"  Code files: {stats['code_files']}")

        return stats

    def load_text_data(self, max_files: Optional[int] = None,
                      min_size: int = 100) -> List[str]:
        """
        Load text data from files.

        Args:
            max_files: Maximum number of files to load (None for all)
            min_size: Minimum file size in bytes

        Returns:
            List of text contents
        """
        print(f"Loading text data...")

        files_to_load = self.text_files[:max_files] if max_files else self.text_files
        loaded_count = 0

        for filepath in files_to_load:
            try:
                # Check file size
                if os.path.getsize(filepath) < min_size:
                    continue

                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        self.loaded_data['text'].append(content)
                        loaded_count += 1

                        if loaded_count % 100 == 0:
                            print(f"  Loaded {loaded_count} text files...")

            except Exception as e:
                print(f"  Warning: Could not read {filepath}: {e}")

        print(f"Loaded {loaded_count} text files")
        return self.loaded_data['text']

    def load_code_data(self, max_files: Optional[int] = None,
                      min_size: int = 50) -> List[str]:
        """
        Load code data from files.

        Args:
            max_files: Maximum number of files to load
            min_size: Minimum file size in bytes

        Returns:
            List of code contents
        """
        print(f"Loading code data...")

        files_to_load = self.code_files[:max_files] if max_files else self.code_files
        loaded_count = 0

        for filepath in files_to_load:
            try:
                # Check file size
                if os.path.getsize(filepath) < min_size:
                    continue

                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        self.loaded_data['code'].append(content)
                        loaded_count += 1

                        if loaded_count % 50 == 0:
                            print(f"  Loaded {loaded_count} code files...")

            except Exception as e:
                print(f"  Warning: Could not read {filepath}: {e}")

        print(f"Loaded {loaded_count} code files")
        return self.loaded_data['code']

    def load_all_data(self, max_text: int = 1000, max_code: int = 500) -> Dict[str, List[str]]:
        """
        Load all available data.

        Args:
            max_text: Maximum text files to load
            max_code: Maximum code files to load

        Returns:
            Dictionary with loaded data by type
        """
        self.scan_directory()

        if self.text_files:
            self.load_text_data(max_files=max_text)

        if self.code_files:
            self.load_code_data(max_files=max_code)

        return self.loaded_data

    def create_sample_dataset(self, output_dir: str, num_samples: int = 100):
        """
        Create a sample dataset for testing.

        Args:
            output_dir: Directory to save sample files
            num_samples: Number of sample files to create
        """
        os.makedirs(output_dir, exist_ok=True)

        # Sample text templates
        text_templates = [
            "The quick brown fox jumps over the lazy dog. This is a sample text for training.",
            "In the beginning, there was nothing but darkness. Then light emerged from the void.",
            "Science and technology have revolutionized our world in countless ways.",
            "The journey of a thousand miles begins with a single step forward.",
            "Knowledge is power, but wisdom is knowing how to use that power effectively.",
        ]

        # Sample code templates
        code_templates = [
            """def hello_world():
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello_world()
""",
            """class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
""",
            """function processData(data) {
    const result = data.map(x => x * 2);
    return result;
}
""",
        ]

        print(f"Creating sample dataset in: {output_dir}")

        # Create text files
        for i in range(num_samples // 2):
            template = random.choice(text_templates)
            filepath = os.path.join(output_dir, f"sample_text_{i}.txt")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(template + f" Sample number {i}.")

        # Create code files
        for i in range(num_samples // 2):
            template = random.choice(code_templates)
            filepath = os.path.join(output_dir, f"sample_code_{i}.py")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(template)

        print(f"Created {num_samples} sample files")

    def split_train_validation(self, validation_ratio: float = 0.1) -> Tuple[Dict, Dict]:
        """
        Split loaded data into training and validation sets.

        Args:
            validation_ratio: Ratio of data to use for validation

        Returns:
            Tuple of (train_data, validation_data)
        """
        train_data = {'text': [], 'code': [], 'math': []}
        val_data = {'text': [], 'code': [], 'math': []}

        for data_type, data_list in self.loaded_data.items():
            if not data_list:
                continue

            # Shuffle
            shuffled = data_list.copy()
            random.shuffle(shuffled)

            # Split
            split_idx = int(len(shuffled) * (1 - validation_ratio))
            train_data[data_type] = shuffled[:split_idx]
            val_data[data_type] = shuffled[split_idx:]

        return train_data, val_data

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about loaded data"""
        return {
            'scanned_files': {
                'text': len(self.text_files),
                'code': len(self.code_files),
                'total': len(self.text_files) + len(self.code_files)
            },
            'loaded_data': {
                'text': len(self.loaded_data['text']),
                'code': len(self.loaded_data['code']),
                'math': len(self.loaded_data['math']),
                'total': sum(len(v) for v in self.loaded_data.values())
            }
        }

