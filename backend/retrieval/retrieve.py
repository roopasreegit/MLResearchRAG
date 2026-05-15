from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


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

all_docs_data=collection.get()
all_docs=all_docs_data["documents"]
all_metadata=all_docs_data["metadatas"]
documents= []
for text,mt in zip(all_docs, all_metadata):
    documents.append(
        Document(
            page_content=text,
            metadata=mt
        )
    )
retriever=BM25Retriever.from_documents(documents)
retriever.k=6


def dense_retrieve_documents(query, top_k=10):

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

def bm25_retrieve(query, top_k=6):
    retriever.k=top_k
    bm25_docs=retriever.invoke(query)

    formatted_results=[]
    for doc in bm25_docs:
        formatted_results.append({
            "text":doc.page_content,
            "metadata":doc.metadata
        })
    return formatted_results

def hybrid_retrieve(query):
    bm25=bm25_retrieve(query, top_k=6)
    dense=dense_retrieve_documents(query, top_k=10)
    combined=bm25+dense

    seen=set()
    unique_res=[]
    for doc in combined:
        if doc["text"] not in seen:
            seen.add(doc["text"])
            unique_res.append(doc)

    return unique_res

#testing
if __name__=="__main__":

    query = "What is Self-RAG?"
    ans=hybrid_retrieve(query)

    print(f"Query: {query}\n")
    print(f"Answer:{ans}")

