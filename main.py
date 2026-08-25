import os
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Create Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Sample RAG data
data = {
    "question": [
        "What is the capital of France?"
    ],
    "answer": [
        "The capital of France is Paris."
    ],
    "contexts": [
        [
            "France is a country in Europe. Paris is the capital of France."
        ]
    ],
}

# Convert to Dataset
dataset = Dataset.from_dict(data)

# Run Ragas evaluation
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
    ],
    llm=llm,
)

print(result)