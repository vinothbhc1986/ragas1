import json
import os
from dotenv import load_dotenv
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import pandas as pd

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._faithfulness import faithfulness

answer_relevancy.strictness = 1

# Load environment variables
load_dotenv()

# Create a Ragas-compatible LLM through Groq's OpenAI-compatible endpoint
llm = llm_factory(
    model="openai/gpt-oss-20b",
    client=OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    ),
)

# Read external CSV file using Pandas
df = pd.read_csv("ragas_10_test_cases.csv")

# Convert contexts from string to Python list
df["contexts"] = df["contexts"].apply(json.loads)

# Convert Pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Sample RAG data
# data = {
#     "question": [
#         "What is the capital of France?"
#     ],
#     "answer": [
#         "The capital of France is Paris."
#     ],
#     "contexts": [
#         [
#             "France is a country in Europe. Paris is the capital of France."
#         ]
#     ],
# }

# Convert to Dataset
# dataset = Dataset.from_dict(data)

# Run Ragas evaluation
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
    ],
    llm=llm,
    embeddings=embeddings,
    column_map={
        "user_input": "question",
        "response": "answer",
        "retrieved_contexts": "contexts",
    },
)

print(result)