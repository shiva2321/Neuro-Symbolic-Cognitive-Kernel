"""
Test script to verify PDF loading functionality
"""

from dashboard_utils.data_loader import DashboardDataLoader, PDF_LIBRARY, PDFMINER_AVAILABLE
from pathlib import Path
import logging
import sys
import os

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s: %(message)s',
    stream=sys.stdout
)

def test_pdf_loading():
    """Test PDF loading with the EAP_ESSAY.pdf file"""
    print(f"\n{'='*70}")
    print(f"ENVIRONMENT DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"PDF Library in use: {PDF_LIBRARY}")
    print(f"PDFMiner available: {PDFMINER_AVAILABLE}")
    
    try:
        import PyPDF2
        print(f"PyPDF2 version: {PyPDF2.__version__}")
    except ImportError:
        print("PyPDF2 not installed")

    loader = DashboardDataLoader()

    # Check if test file exists
    test_file = Path("uploads/EAP_ESSAY.pdf")

    if not test_file.exists():
        print(f"\n❌ Test file not found: {test_file}")
        print("Please upload a PDF file first through the dashboard.")
        return

    # Check file size
    file_size = test_file.stat().st_size
    print(f"\nFile: {test_file.name}")
    print(f"Size: {file_size} bytes ({file_size/1024:.2f} KB)")

    if file_size == 0:
        print("❌ File is empty (0 bytes)")
        return

    # Check header and footer
    try:
        with open(test_file, 'rb') as f:
            header = f.read(10)
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 20), os.SEEK_SET)
            footer = f.read()

        print(f"Header: {header}")
        print(f"Footer: {footer}")

        if not header.startswith(b'%PDF-'):
            print("❌ File does not start with %PDF- marker")

        if b'%%EOF' not in footer:
            print("❌ File does not end with %%EOF marker (likely truncated/corrupted)")

    except Exception as e:
        print(f"Error reading file: {e}")

    print(f"\n{'='*70}")
    print(f"Testing PDF Loader with: {test_file.name}")
    print(f"{'='*70}\n")

    # Load the PDF
    chunks = loader.load_file(str(test_file))

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Total chunks extracted: {len(chunks)}")

    if len(chunks) > 0:
        print(f"\n✓ Successfully extracted text from PDF!")
        print(f"\nFirst 5 chunks:")
        for i, chunk in enumerate(chunks[:5], 1):
            print(f"\n{i}. {chunk[:100]}..." if len(chunk) > 100 else f"\n{i}. {chunk}")
    else:
        print(f"\n❌ No text extracted from PDF")
        # Try to debug page content
        try:
            import PyPDF2
            with open(test_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                print(f"Number of pages: {len(reader.pages)}")
                if len(reader.pages) > 0:
                    page = reader.pages[0]
                    text = page.extract_text()
                    print(f"Page 1 raw text length: {len(text) if text else 0}")
                    if text:
                        print(f"Page 1 raw text snippet: {text[:100]!r}")
        except Exception as e:
            print(f"Debug extraction failed: {e}")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    test_pdf_loading()
