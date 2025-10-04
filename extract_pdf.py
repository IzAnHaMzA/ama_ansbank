import sys
import os

# Try different PDF extraction methods
try:
    import PyPDF2
    def extract_with_pypdf2(pdf_path):
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    PDF_METHOD = "PyPDF2"
except ImportError:
    try:
        import fitz  # PyMuPDF
        def extract_with_fitz(pdf_path):
            text = ""
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            return text
        PDF_METHOD = "PyMuPDF"
    except ImportError:
        print("No PDF library available. Please install PyPDF2 or PyMuPDF")
        sys.exit(1)

def extract_pdf_text(pdf_path):
    if PDF_METHOD == "PyPDF2":
        return extract_with_pypdf2(pdf_path)
    else:
        return extract_with_fitz(pdf_path)

if __name__ == "__main__":
    pdf_files = [
        "AMA Question Bank Unit test 2.pdf",
        "Unit - IV Types of Learning_AMA.pdf", 
        "Unit_3_Introduction_To_ML.pdf"
    ]
    
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            print(f"\n{'='*50}")
            print(f"EXTRACTING: {pdf_file}")
            print(f"{'='*50}")
            try:
                text = extract_pdf_text(pdf_file)
                print(text[:2000])  # First 2000 characters
                print(f"\n... (showing first 2000 chars of {len(text)} total)")
                
                # Save full text to file
                output_file = pdf_file.replace('.pdf', '_extracted.txt')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Full text saved to: {output_file}")
            except Exception as e:
                print(f"Error extracting {pdf_file}: {e}")
        else:
            print(f"File not found: {pdf_file}")
