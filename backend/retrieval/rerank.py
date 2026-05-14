import os
from dotenv import load_dotenv
import cohere

from backend.retrieval.retrieve import hybrid_retrieve

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

co = cohere.ClientV2(api_key=COHERE_API_KEY)

def rerank_documents(query, retrieved_docs, top_n=3):

    documents = [
        doc["text"]
        for doc in retrieved_docs
    ]

    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=top_n
    )

    reranked_results = []

    for item in response.results:

        original_doc = retrieved_docs[item.index]

        reranked_results.append({
            "text": original_doc["text"],
            "metadata": original_doc["metadata"],
            "relevance_score": item.relevance_score
        })

    return reranked_results

#for testing

if __name__ == "__main__":

    query = "How do modern architectures improve transformer efficiency?"

    retrieved_docs = hybrid_retrieve(query=query)

    reranked_docs = rerank_documents(
        query=query,
        retrieved_docs=retrieved_docs,
        top_n=3
    )

    print("\nTOP RERANKED RESULTS\n")

    for i, doc in enumerate(reranked_docs):

        print("=" * 60)

        print(f"RANK {i+1}")
        print(f"SCORE: {doc['relevance_score']}")

        print("\nPAPER:")
        print(doc["metadata"]["paper_title"])

        print("\nTEXT:")
        print(doc["text"][:1000])

        print("\n")