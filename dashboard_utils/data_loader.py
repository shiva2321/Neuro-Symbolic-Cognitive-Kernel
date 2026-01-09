"""
Data Loader for NCGN Dashboard
Handles file uploads and data processing.
"""

try:
    import PyPDF2
    PDF_LIBRARY = 'PyPDF2'
except ImportError:
    try:
        import pypdf as PyPDF2
        PDF_LIBRARY = 'pypdf'
    except ImportError:
        PyPDF2 = None
        PDF_LIBRARY = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    docx = None
    DOCX_AVAILABLE = False

from pathlib import Path
from typing import List, Dict, Any
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardDataLoader:
    """
    Loads and processes various file formats for training.
    """

    def __init__(self):
        """Initialize data loader"""
        self.supported_formats = ['.txt', '.py', '.java', '.cpp', '.js']

        # Add PDF support if library is available
        if PyPDF2 is not None:
            self.supported_formats.append('.pdf')
            logger.info(f"PDF support enabled using {PDF_LIBRARY}")
        else:
            logger.warning("PyPDF2/pypdf not available. PDF files will not be supported. Install with: pip install PyPDF2")

        if PDFMINER_AVAILABLE:
            logger.info("PDFMiner support enabled (fallback)")
            if '.pdf' not in self.supported_formats:
                self.supported_formats.append('.pdf')

        # Add DOCX support if library is available
        if DOCX_AVAILABLE:
            self.supported_formats.append('.docx')
            logger.info("DOCX support enabled")
        else:
            logger.warning("python-docx not available. DOCX files will not be supported. Install with: pip install python-docx")

    def load_file(self, file_path: str) -> List[str]:
        """
        Load file and return text content.

        Args:
            file_path: Path to file

        Returns:
            List of text chunks
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported format: {suffix}")

        if suffix == '.txt' or suffix in ['.py', '.java', '.cpp', '.js']:
            return self.load_text(file_path)
        elif suffix == '.pdf':
            return self.load_pdf(file_path)
        elif suffix == '.docx':
            return self.load_docx(file_path)

        return []

    def load_text(self, file_path: Path) -> List[str]:
        """Load text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split into sentences/chunks
            chunks = text.split('\n')
            chunks = [c.strip() for c in chunks if c.strip()]

            logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
            return chunks

        except Exception as e:
            logger.error(f"Error loading text file: {e}")
            return []

    def load_pdf(self, file_path: Path) -> List[str]:
        """Load PDF file"""
        chunks = []
        
        # Check if file is actually a PDF
        try:
            with open(file_path, 'rb') as f:
                header = f.read(5)
                if header != b'%PDF-':
                    logger.error(f"Invalid PDF file: Header is {header} instead of %PDF-")
                    if header.startswith(b'\xac\xed'):
                        logger.error("File appears to be a Java Serialized Object, not a PDF.")
                    return []
        except Exception as e:
            logger.error(f"Error reading file header: {e}")
            return []
        
        # Method 1: PyPDF2/pypdf
        if PyPDF2 is not None:
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    
                    # Handle encryption
                    if pdf_reader.is_encrypted:
                        try:
                            pdf_reader.decrypt('')
                            logger.info("PDF decrypted with empty password")
                        except Exception:
                            logger.warning("PDF is encrypted and could not be decrypted with empty password")

                    logger.info(f"PDF has {len(pdf_reader.pages)} pages")

                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            text = page.extract_text()
                            
                            # Try layout mode if default fails (pypdf feature)
                            if not text:
                                try:
                                    text = page.extract_text(extraction_mode="layout")
                                except:
                                    pass

                            if text:
                                # Clean text
                                text = text.strip()
                                
                                # Split strategies
                                # 1. Split by newlines
                                lines = [line.strip() for line in text.split('\n') if line.strip()]
                                
                                # 2. Split by sentences if lines are too long or few
                                if len(lines) < 3 or any(len(l) > 200 for l in lines):
                                    sentences = re.split(r'[.!?]+\s+', text)
                                    sentences = [s.strip() for s in sentences if s.strip()]
                                    if len(sentences) > len(lines):
                                        chunks.extend(sentences)
                                    else:
                                        chunks.extend(lines)
                                else:
                                    chunks.extend(lines)
                                    
                            else:
                                logger.debug(f"Page {page_num + 1}: no text extracted")
                                if hasattr(page, 'images') and page.images:
                                    logger.warning(f"Page {page_num + 1} has images but no text. OCR required.")
                        except Exception as e:
                            logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                            continue
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed: {e}")

        # Method 2: PDFMiner (Fallback)
        if not chunks and PDFMINER_AVAILABLE:
            logger.info("Falling back to PDFMiner...")
            try:
                text = pdfminer_extract_text(file_path)
                if text:
                    chunks = [line.strip() for line in text.split('\n') if line.strip()]
                    logger.info(f"PDFMiner extracted {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"PDFMiner extraction failed: {e}")

        # Filter chunks
        chunks = [c for c in chunks if len(c) > 3]
        
        if chunks:
            logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
        else:
            logger.error(f"No text extracted from PDF {file_path.name}. It may be image-based or empty.")

        return chunks

    def load_docx(self, file_path: Path) -> List[str]:
        """Load DOCX file"""
        if not DOCX_AVAILABLE:
            logger.error("python-docx not available. Install with: pip install python-docx")
            return []

        try:
            doc = docx.Document(file_path)
            chunks = []

            for para in doc.paragraphs:
                if para.text.strip():
                    chunks.append(para.text.strip())

            logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
            return chunks

        except Exception as e:
            logger.error(f"Error loading DOCX: {e}")
            return []

    def batch_load(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Load multiple files.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping filename to chunks
        """
        results = {}

        for path in file_paths:
            try:
                chunks = self.load_file(path)
                results[Path(path).name] = chunks
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")

        return results
