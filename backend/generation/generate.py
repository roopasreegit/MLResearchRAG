import os
from dotenv import load_dotenv
from backend.retrieval.retrieve import retrieve_documents
from backend.retrieval.rerank import rerank_documents

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

llm= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature = 0.3
)

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template ="""
You are an AI research assistant.

Answer the user's question ONLY using the provided context.

If the answer is not present in the context,
say:
"I could not find enough information in the retrieved papers."

Be concise, accurate, and grounded.

Also mention which paper the answer came from when possible.

==================== CONTEXT ====================

{context}

=================================================

Question:
{query}

Answer:
"""
)

def build_context(reranked_docs):

    context = ""

    for i, doc in enumerate(reranked_docs):

        paper_title = doc["metadata"]["paper_title"]

        text = doc["text"]

        context += f"""
SOURCE {i+1}: {paper_title}

{text}

--------------------------------------------------

"""

    return context


#main rag pipeline

def generate_answer(query):

    retrieved_docs = retrieve_documents(
        query=query,
        top_k=10
    )

    reranked_docs = rerank_documents(
        query=query,
        retrieved_docs=retrieved_docs,
        top_n=3
    )

    context = build_context(reranked_docs)

    final_prompt = prompt_template.format(
        context=context,
        query=query
    )

    response = llm.invoke(final_prompt)

    return {
        "query": query,
        "answer": response.content,
        "contexts": [
            doc["text"]
            for doc in reranked_docs
        ],
        "sources": [
            doc["metadata"]["paper_title"]
            for doc in reranked_docs
        ]
    }


#testing

if __name__ == "__main__":

    query = "How does Self-RAG improve factuality?"

    result = generate_answer(query)

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(result["query"])

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in result["sources"]:
        print(source)