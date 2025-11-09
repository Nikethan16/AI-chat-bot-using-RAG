import requests
from config.config import GOOGLE_API_KEY, GOOGLE_CX_ID, GOOGLE_SEARCH_URL

def google_search(query, num_results=3):
    try:
        params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX_ID, "q": query, "num": num_results}
        res = requests.get(GOOGLE_SEARCH_URL, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        results = []
        for item in data.get("items", []):
            results.append(f"{item['title']}\n{item['snippet']}\n{item['link']}")
        return "\n\n".join(results[:num_results])
    except Exception as e:
        return f"Web search failed: {e}"
