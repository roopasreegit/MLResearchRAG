import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from load_pdfs import load_and_chunk_pdfs

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"



client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    client.delete_collection("research_papers")
    print("Old collection deleted.")
except:
    print("No existing collection found.")

print("Loading and chunking PDFs...\n")
all_chunks = load_and_chunk_pdfs()
print(f"Total chunks received: {len(all_chunks)}\n")


print("Loading embedding model...\n")
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded.\n")

collection = client.get_or_create_collection(name="research_papers", configuration={"hnsw": {"space":"cosine", "ef_construction": 200}})

documents = []
metadatas = []
ids = []
for chunk in all_chunks:
    documents.append(chunk["text"])
    metadatas.append(chunk["metadata"])
    ids.append(chunk["metadata"]["chunk_id"])

print("Creating embeddings...\n")
embeddings = embedding_model.encode(documents)
print("Embeddings created.\n")

collection.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    documents=documents,
    metadatas=metadatas
)

print("=" * 50)
print("SUCCESSFULLY STORED IN CHROMADB")
print("=" * 50)

print(f"Total stored chunks: {len(documents)}")