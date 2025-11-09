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
    Includes:
      - Strict healthcare-only filtering
      - Automatic web search fallback if context insufficient
      - Context-aware, structured, and clear medical responses
    """

    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in environment variables!")

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    # ===============================
    # Style configuration
    # ===============================
    if response_mode.lower() == "concise":
        style_note = (
            "Provide a medically accurate response of about 80–120 words. "
            "Avoid fluff. Focus only on the key facts and practical advice."
        )
    else:
        style_note = (
            "Write a detailed, structured explanation of about 200–300 words. "
            "Include definition, causes or mechanisms, treatment insights, "
            "and preventive steps when appropriate. Keep tone factual and calm."
        )

    # ===============================
    # Source context
    # ===============================
    source_note = ""
    if sources:
        source_note = f"\n\nRelevant Sources: {', '.join(sources)}"

    # ===============================
    # System-level instruction
    # ===============================
    system_prompt = """
You are a professional and responsible AI healthcare assistant.
You are trained exclusively in health, medicine, nutrition, diagnostics, and wellness.

Rules:
1. Only respond to healthcare-related queries.
2. If a question is unrelated to healthcare (e.g., politics, sports, technology, finance, celebrities),
   reply strictly: "I'm designed to answer healthcare-related questions only."
3. If the context or data provided is insufficient to give a confident answer, respond:
   "I don’t have enough medical information to answer confidently."
4. Use evidence-based reasoning; never guess or fabricate.
5. Maintain clarity, empathy, and professionalism.
6. Never include disclaimers like 'This is not medical advice' — a disclaimer is already shown in the UI.
"""

    # ===============================
    # User + context prompt
    # ===============================
    full_prompt = f"""
### Context:
{context or "No relevant context retrieved."}

### User Question:
{query}

### Response Guidelines:
{style_note}

{source_note}

### Instructions:
- If healthcare-related → give factual, clear response.
- If unrelated → politely refuse as per rules.
- If context is insufficient → say you don’t have enough info.
- Avoid speculation and unnecessary verbosity.
- Output should be human-readable and structured.
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": full_prompt.strip()},
        ],
        "max_tokens": 900,
        "temperature": LLM_TEMPERATURE,
    }

    # ===============================
    # First API call (with RAG context)
    # ===============================
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()

        # Check for insufficient info phrase → trigger web fallback
        if any(
            phrase.lower() in answer.lower()
            for phrase in [
                "i don’t have enough medical information",
                "i don't have enough medical information",
                "insufficient context",
                "not enough information"
            ]
        ):
            print("⚠️ Insufficient context detected — performing web search fallback...")
            web_context = google_search(query)
            if web_context and web_context.strip():
                # Rerun prompt with web context
                web_prompt = f"{full_prompt}\n\n### Additional Web Search Results:\n{web_context}"
                payload["messages"][1]["content"] = web_prompt
                response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                answer = data["choices"][0]["message"]["content"].strip()
                sources = (sources or []) + ["Google Search"]

        return answer

    except requests.exceptions.RequestException as e:
        print("Request Error:", e)
        if hasattr(e, "response") and e.response is not None:
            print("Response:", e.response.text)
        return "⚠️ Unable to generate response at the moment."

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print("Parsing Error:", e)
        return "⚠️ Received invalid response from the model."
