from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "chroma_db"

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)



client = chromadb.PersistentClient(path=str(CHROMA_DIR))

print(client.list_collections())

collection = client.get_collection(
    name="research_papers"
)
print("Connected to ChromaDB.\n")

#query = "What is Self-RAG?"
#print(f"Query: {query}\n")

def retrieve_documents(query, top_k=10):

    query_embedding = embedding_model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    retrieved_docs = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]

    formatted_results = []

    for doc, metadata in zip(
        retrieved_docs,
        retrieved_metadata
    ):

        formatted_results.append({
            "text": doc,
            "metadata": metadata
        })

    return formatted_results