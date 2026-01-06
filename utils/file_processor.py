"""
File Processor Utility
Handles extraction of text from various file formats (PDF, DOC, DOCX, etc.)
"""

import os
from typing import Optional


class FileProcessor:
    """Process various file formats and extract text content"""

    def __init__(self):
        self.pdf_available = False
        self.docx_available = False

        # Try to import PDF libraries
        try:
            import PyPDF2
            self.pdf_available = True
        except ImportError:
            print("Warning: PyPDF2 not available. PDF processing disabled.")

        # Try to import DOCX libraries
        try:
            import docx
            self.docx_available = True
        except ImportError:
            print("Warning: python-docx not available. DOCX processing disabled.")

    def process_file(self, filepath: str) -> Optional[str]:
        """
        Process a file and extract text content.

        Args:
            filepath: Path to the file

        Returns:
            Extracted text content or None if processing failed
        """
        if not os.path.exists(filepath):
            return None

        ext = filepath.lower().rsplit('.', 1)[-1]

        try:
            if ext == 'pdf':
                return self.process_pdf(filepath)
            elif ext in ['docx', 'doc']:
                return self.process_docx(filepath)
            elif ext in ['txt', 'text', 'md']:
                return self.process_text(filepath)
            elif ext in ['py', 'js', 'java', 'cpp', 'c', 'cs', 'rb', 'go', 'rs', 'html', 'css', 'json', 'xml']:
                return self.process_text(filepath)
            else:
                # Try to process as text
                return self.process_text(filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None

    def process_pdf(self, filepath: str) -> Optional[str]:
        """Extract text from PDF file"""
        if not self.pdf_available:
            return None

        try:
            import PyPDF2

            text = []
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)

            return '\n'.join(text)
        except Exception as e:
            print(f"Error processing PDF {filepath}: {e}")
            return None

    def process_docx(self, filepath: str) -> Optional[str]:
        """Extract text from DOCX file"""
        if not self.docx_available:
            return None

        try:
            import docx

            doc = docx.Document(filepath)
            text = []

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)

            return '\n'.join(text)
        except Exception as e:
            print(f"Error processing DOCX {filepath}: {e}")
            return None

    def process_text(self, filepath: str) -> Optional[str]:
        """Extract text from plain text file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            print(f"Error processing text file {filepath}: {e}")
            return None

    def batch_process(self, directory: str) -> dict:
        """
        Process all files in a directory.

        Args:
            directory: Directory containing files

        Returns:
            Dictionary mapping filenames to extracted text
        """
        results = {}

        if not os.path.exists(directory):
            return results

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)

            if os.path.isfile(filepath):
                text = self.process_file(filepath)
                if text:
                    results[filename] = text

        return results

