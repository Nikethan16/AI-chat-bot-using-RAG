from utils.rag_search import get_relevant_context
from models.llm import generate_answer
from config.config import CHUNKS_PATH

query = "What are the main causes and prevention strategies for hypertension?"
context = get_relevant_context(query, k=5)

print("🧠 Sending to LLM...\n")
response = generate_answer(query, context, response_mode="detailed")

print("💬 Model Response:\n")
print(response)
