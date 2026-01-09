import sys
print("Checking OCR capabilities...")

try:
    import pytesseract
    print(f"pytesseract: {pytesseract.__version__}")
except ImportError:
    print("pytesseract: Not installed")

try:
    import pdf2image
    print("pdf2image: Installed")
except ImportError:
    print("pdf2image: Not installed")

try:
    import PIL
    print(f"Pillow: {PIL.__version__}")
except ImportError:
    print("Pillow: Not installed")
