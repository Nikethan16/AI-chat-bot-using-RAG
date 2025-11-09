from utils.rag_search import get_relevant_context

query = "What are the symptoms and prevention methods of hypertension?"
context = get_relevant_context(query, k=5)

print("🔎 Retrieved Context:\n")
print(context[:1200])
