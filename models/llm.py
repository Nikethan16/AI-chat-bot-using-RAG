import os
import requests
import json
from utils.web_search import google_search
from config.config import (
    GROQ_API_KEY,
    GROQ_API_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

def generate_answer(query: str, context: str, response_mode: str = "detailed", sources: list = None):
    """
    Generate grounded, healthcare-focused answers using Groq API (LLaMA 3.3 70B).
    Features:
      - Strict healthcare-only domain filtering
      - Automatic web search fallback if context is insufficient
      - Clear, structured, medically factual responses
    """

    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in environment variables.")

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    # RESPONSE STYLE CONFIGURATION
    if response_mode.lower() == "concise":
        style_note = (
            "Provide a medically accurate response of about 80–120 words. "
            "Avoid fluff; focus only on relevant and practical facts."
        )
    else:
        style_note = (
            "Write a structured, detailed explanation of about 200–300 words. "
            "Include definition, causes, treatments, and prevention tips when appropriate."
        )

    # SOURCE CONTEXT
    source_note = f"\n\nRelevant Sources: {', '.join(sources)}" if sources else ""

    # SYSTEM PROMPT
    system_prompt = """
You are a professional AI healthcare assistant trained exclusively in medicine, nutrition, diagnostics, and wellness.

Rules:
1. Only respond to healthcare-related questions.
2. If the query is unrelated (e.g., sports, politics, technology, etc.), reply:
   "I'm designed to answer healthcare-related questions only."
3. If the context is insufficient, respond:
   "I don’t have enough medical information to answer confidently."
4. Never fabricate or guess — rely only on verified data.
5. Keep tone factual, calm, and empathetic.
6. Never include disclaimers — one is already displayed in the app UI.
"""

    # COMBINE CONTEXT AND QUERY
    full_prompt = f"""
### Context:
{context or "No relevant context available."}

### User Question:
{query}

### Response Guidelines:
{style_note}

{source_note}

### Instructions:
- If healthcare-related → give factual, clear info.
- If unrelated → politely refuse.
- If context insufficient → say so directly.
- Keep language simple and structured.
"""

    # PAYLOAD CREATION
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": full_prompt.strip()},
        ],
        "max_tokens": 900,
        "temperature": LLM_TEMPERATURE,
    }

    # PRIMARY REQUEST (RAG Context)
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()

        # FALLBACK: WEB SEARCH IF CONTEXT TOO WEAK
        insufficient_phrases = [
            "i don’t have enough medical information",
            "i don't have enough medical information",
            "insufficient context",
            "not enough information",
        ]

        if any(phrase in answer.lower() for phrase in insufficient_phrases):
            web_context = google_search(query)
            if web_context and web_context.strip():
                combined_prompt = f"{full_prompt}\n\n### Additional Web Search Results:\n{web_context}"
                payload["messages"][1]["content"] = combined_prompt

                try:
                    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    answer = data["choices"][0]["message"]["content"].strip()
                    sources = (sources or []) + ["Google Search"]
                except Exception:
                    answer += "\n\n⚠️ Web search data could not be processed."

        return answer

    except requests.exceptions.RequestException:
        return "⚠️ Network or API request failed while generating the response."

    except (KeyError, IndexError, json.JSONDecodeError):
        return "⚠️ Received an invalid response from the model."
