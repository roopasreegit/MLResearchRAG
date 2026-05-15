from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY=os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

def rewrite_query(query):
    prompt=f"""
You are an AI Research Assistant specialised in Gen AI, LLMs, RAG and
fine-tuning based state of the art research papers.

Rewrite the user query into a technically detailed concise query optimized
for retrieving: GenAI, RAG, Fine tuning, and LLM related Research papers such as:

Attention Is All You Need
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Improving Language Understanding by Generative Pre-Training
Mistral 7B
Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
Learning to Summarize with Human Feedback
LoRA: Low-Rank Adaptation of Large Language Models
QLoRA: Efficient Finetuning of Quantized LLMs
Distilling the Knowledge in a Neural Network
LLM in a flash: Efficient Large Language Model Inference with Limited Memory
Retentive Network: A Successor to Transformer for Large Language Models
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
Corrective Retrieval Augmented Generation
Lost in the Middle: How Language Models Use Long Contexts
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
RAG vs Fine Tuning

User query: {query}

return ONLY the rewritten query optimized for RAG retrieval.
"""
    response=llm.invoke(prompt)
    return response.content.strip()


#testing
if __name__=="__main__":
    q="I am looking into how to make language models more reliable when they encounter tasks requiring factual precision. Should I focus on changing how the model processes information during its actual deployment, or is it better to modify the underlying structural parameters before it ever runs?"

    query=rewrite_query(q)
    print(f"Query:{query}")