import requests
from config.config import GOOGLE_API_KEY, GOOGLE_CX_ID, GOOGLE_SEARCH_URL

def google_search(query, num_results=3):
    """
    Perform a Google Custom Search and return short snippets of web results.
    Used as a fallback when the RAG system lacks relevant context.
    """
    try:
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX_ID,
            "q": query,
            "num": num_results
        }

        response = requests.get(GOOGLE_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("items", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            results.append(f"{title}\n{snippet}\n{link}")

        return "\n\n".join(results[:num_results])

    except Exception as e:
        # Keep logs for debugging but don’t show errors to the user
        print(f"[Web Search Error] {e}")
        return ""
