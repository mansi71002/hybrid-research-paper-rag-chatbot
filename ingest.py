from pypdf import PdfReader
import pdfplumber

def extract_text(pdf_path):
    text = ""

    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except:
        pass

    if len(text.strip()) < 500:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    return text
