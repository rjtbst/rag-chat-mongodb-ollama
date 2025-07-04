import os
import json
import fitz  # for PDFs
import docx  # for DOCX files python-docx
from bs4 import BeautifulSoup
from pathlib import Path

def load_txt(file_path):
    return Path(file_path).read_text(encoding="utf-8")

def load_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def load_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_md(file_path):
    return Path(file_path).read_text(encoding="utf-8")

def load_html(file_path):
    html = Path(file_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n")

def load_json(file_path):
    try:
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error loading JSON: {e}"

def load_docs_from_folder(folder):
    all_docs = []
    loaded_files = []
    failed_files = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        lower_file = file.lower()

        try:
            if lower_file.endswith(".txt"):
                content = load_txt(path)
            elif lower_file.endswith(".pdf"):
                content = load_pdf(path)
            elif lower_file.endswith(".docx"):
                content = load_docx(path)
            elif lower_file.endswith(".md"):
                content = load_md(path)
            elif lower_file.endswith(".html"):
                content = load_html(path)
            elif lower_file.endswith(".json"):
                content = load_json(path)
            else:
                print(f"❌ Unsupported file type skipped: {file}")
                failed_files.append(file)
                continue

            all_docs.append({"filename": file, "content": content})
            loaded_files.append(file)
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
            failed_files.append(file)

    print("\n✅ Summary:")
    print(f"Loaded files ({len(loaded_files)}): {loaded_files}")
    print(f"Failed/skipped files ({len(failed_files)}): {failed_files}")
    return all_docs


