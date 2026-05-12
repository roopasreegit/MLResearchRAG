from pathlib import Path
import pymupdf4llm
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "data" / "papers" 

#md_text = pymupdf4llm.to_markdown(str(pdf_files[0]))
#print(md_text[:1000])

def clean_text(text: str) -> str:

    text = text.replace("\t", " ")

    # Remove excessive spaces
    text = re.sub(r" +", " ", text)

    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove weird unicode null chars if present
    text = text.replace("\x00", "")

    return text.strip()

#chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=[
        "\n# ",     # markdown headings
        "\n## ",
        "\n\n",     # paragraphs
        "\n",       # lines
        ". ",       # sentences
        " ",        # words
        ""          # fallback
    ]
)

def load_and_chunk_pdfs():

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))

    print(f"\nFound {len(pdf_files)} PDFs.\n")

    all_chunks = []

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}")

        raw_text = pymupdf4llm.to_markdown(str(pdf_path))

        cleaned_text = clean_text(raw_text)

        chunks = text_splitter.split_text(cleaned_text)

        print(f"Created {len(chunks)} chunks.\n")

        for idx, chunk in enumerate(chunks):

            chunk_data = {
                "text": chunk,
                "metadata": {
                    "paper_title": pdf_path.stem,
                    "source_file": pdf_path.name,
                    "chunk_id": f"{pdf_path.stem}_chunk_{idx}",
                    "chunk_index": idx
                }
            }

            all_chunks.append(chunk_data)

    return all_chunks