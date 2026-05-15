import os
from dotenv import load_dotenv
from backend.retrieval.retrieve import hybrid_retrieve
from backend.retrieval.rerank import rerank_documents
from backend.retrieval.query_rewrite import rewrite_query
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

llm= ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature = 0.5
)

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template ="""
You are an AI Research Assistant specialised in Gen AI, LLMs, RAG and
fine-tuning based state of the art research papers.


Answer the user's question using the provided context in an information rich manner with technical details.

If the answer is not present in the context,
say:
"I could not find enough information in the retrieved papers."

Be concise, accurate, and grounded.

Also mention which paper the answer came from.

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

    retrieved_docs = hybrid_retrieve(query=query)

    reranked_docs = rerank_documents(query=query,retrieved_docs=retrieved_docs,top_n=5)

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

def rewrite_before_gen(query):
    q = rewrite_query(query)
    generate_answer(q)


#testing

if __name__ == "__main__":

    initial_query = "I am looking into how to make language models more reliable when they encounter tasks requiring factual precision. Should I focus on changing how the model processes information during its actual deployment, or is it better to modify the underlying structural parameters before it ever runs?"
    query=rewrite_query(initial_query)

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