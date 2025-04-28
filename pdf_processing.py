import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_file_path):
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF not found: {pdf_file_path}")
    
    text = ""
    try:
        with fitz.open(pdf_file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")
    
    if not text.strip():
        raise ValueError("Extracted text is empty - the PDF might be image-based")
    
    return text