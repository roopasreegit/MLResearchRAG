from dotenv import load_dotenv
import os
import time

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)

from deepeval.models import GeminiModel

from backend.retrieval.query_rewrite import rewrite_query
from backend.evaluation.eval_dataset import eval_data
from backend.generation.generate import generate_answer

load_dotenv()

evaluation_model = GeminiModel(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

test_cases = []

for item in eval_data[:1]:

    question = item["question"]
    ground_truth = item["ground_truth"]

    initial_query = question
    query=rewrite_query(initial_query)

    result = generate_answer(query)

    time.sleep(25)

    test_case = LLMTestCase(
        input=question,
        actual_output=result["answer"],
        expected_output=ground_truth,
        retrieval_context=result["contexts"]
    )

    test_cases.append(test_case)

metrics = [
    #AnswerRelevancyMetric(model=evaluation_model, async_mode=False)
    #FaithfulnessMetric(model=evaluation_model, async_mode=False),
    #ContextualPrecisionMetric(model=evaluation_model, async_mode=False),
    ContextualRecallMetric(model=evaluation_model, async_mode=False)
]

evaluate(
    test_cases=test_cases,
    metrics=metrics
)

#ragas stuff that didnt work cos no free openapi key
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import(faithfulness, context_precision, context_recall, answer_relevancy)

from backend.evaluation.eval_dataset import eval_data
from backend.generation.generate import generate_answer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper

import os
from dotenv import load_dotenv

load_dotenv()

questions = []
answers = []
contexts = []
ground_truths = []

for item in eval_data:
    question=item["question"]
    ground_truth=item["ground_truth"]
    result=generate_answer(question)
    questions.append(question)
    answers.append(result["answer"])
    contexts.append(result["contexts"])
    ground_truths.append(ground_truth)

dataset=Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

base_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

evaluator_llm = LangchainLLMWrapper(base_llm)

evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

result = evaluate(
    dataset=dataset,

     metrics=[

        faithfulness,

        answer_relevancy,

        context_precision,

        context_recall
    ]
)

print("\n")
print("RAGAS EVALUATION RESULT")
print(result)
"""