"""pdfx - self-hosted hybrid PDF extraction.

Pipeline shape:
    detect native text layer per page
      -> native pages parsed directly (Docling or PyMuPDF)
      -> image/scanned pages sent to a local GLM-OCR served by Ollama
      -> numeric fields cross-checked deterministically before output
"""

__version__ = "0.1.0"
