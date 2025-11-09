import streamlit as st
import os, time, json
from utils.rag_search import get_relevant_context
from models.llm import generate_answer
from utils.web_search import google_search
from utils.pdf_parser import extract_text_from_pdf
from utils.chunking import chunk_text
from models.embeddings import build_faiss_index
from config.config import ENABLE_WEB_SEARCH

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Healthcare Assistant", page_icon="💊", layout="centered")

# ===============================
# HEADER
# ===============================
st.markdown("<h1 style='text-align:center; color:white;'>Healthcare Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8;'>Powered by RAG + Verified Medical Knowledge Base</p>", unsafe_allow_html=True)

# ===============================
# SESSION STATE
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "response_mode" not in st.session_state:
    st.session_state.response_mode = "Concise"
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "uploaded_texts" not in st.session_state:
    st.session_state.uploaded_texts = []
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "source_used" not in st.session_state:
    st.session_state.source_used = None

# ===============================
# CHAT HISTORY
# ===============================
st.markdown("<div style='max-height:68vh; overflow-y:auto; padding-bottom:6rem;'>", unsafe_allow_html=True)
for msg in st.session_state.messages[-8:]:
    if msg["role"] == "user":
        st.markdown(f"**🧍 You:** {msg['content']}")
    else:
        st.markdown(f"<div class='response-box'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
        if "sources" in msg and msg["sources"]:
            src_list = ", ".join(msg["sources"])
            st.markdown(
                f"<small style='color:#94a3b8; display:block; margin-top:-0.6rem; text-align:right;'>"
                f"🧠 Sources: {src_list}</small>",
                unsafe_allow_html=True
            )
st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# CHAT CONTROLS (Toggle + Input + Upload)
# ===============================
st.markdown("<div style='position:fixed; bottom:0; left:0; right:0; background-color:#0b0c10; padding:0.8rem 2rem; z-index:10;'>", unsafe_allow_html=True)

col_toggle = st.container()
with col_toggle:
    detailed_mode = st.toggle("Detailed", value=False, key="detail_toggle")
st.session_state.response_mode = "Detailed" if detailed_mode else "Concise"

upload_col, input_col = st.columns([0.08, 0.92])
with upload_col:
    if st.button("📎", key="upload_button", help="Upload healthcare document"):
        st.session_state.show_uploader = not st.session_state.show_uploader
with input_col:
    user_query = st.chat_input("Ask a medical question...")

st.markdown("""
<div style='text-align:center; color:#cbd5e1; font-size:0.85rem; padding:0.8rem;'>
<i style='color:#00a6a6;'>ⓘ</i> This assistant provides educational information only. Consult qualified professionals for medical advice.
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# FILE UPLOAD HANDLER
# ===============================
if st.session_state.show_uploader:
    st.markdown("<h4 style='text-align:center;'>📄 Upload Healthcare Documents</h4>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="multi_file_uploader"
    )

    if uploaded_files:
        save_dir = "data/raw_pdfs"
        os.makedirs(save_dir, exist_ok=True)
        st.session_state.uploaded_texts.clear()

        for uploaded_file in uploaded_files:
            path = os.path.join(save_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if uploaded_file.name not in st.session_state.uploaded_docs:
                st.session_state.uploaded_docs.append(uploaded_file.name)

            st.success(f"✅ Uploaded: {uploaded_file.name}")

            try:
                text = extract_text_from_pdf(path)
                if text.strip():
                    st.session_state.uploaded_texts.append(text)
                    st.write(f"🧠 Extracted text from {uploaded_file.name}")
                else:
                    st.warning(f"⚠️ No readable text found in {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")

        if st.session_state.uploaded_texts:
            st.success("✅ All documents processed and ready for insights!")
            if st.button("🩺 Generate Combined Insights"):
                combined_text = "\n\n".join(st.session_state.uploaded_texts)
                with st.spinner("Analyzing all uploaded documents..."):
                    context, sources = get_relevant_context(
                        "combined medical report analysis",
                        k=8,
                        report_text=combined_text
                    )
                    if len(context.strip()) < 200 and ENABLE_WEB_SEARCH:
                        context = google_search("general medical report insights")
                        sources = ["Google Search"]
                        st.session_state.source_used = "Web Search"
                    else:
                        st.session_state.source_used = "RAG"

                    combined_insights = generate_answer(
                        query="Summarize insights across all uploaded health documents.",
                        context=context,
                        response_mode="Detailed",
                        sources=sources
                    )
                    st.markdown(
                        f"<div style='background-color:#1f2937; border-radius:14px; padding:1rem; margin:1rem 0;'>{combined_insights}</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"🧠 Source: {st.session_state.source_used} | References: {', '.join(sources)}")

if st.session_state.uploaded_docs:
    st.markdown(
        f"<p style='text-align:center; color:#94a3b8; font-size:0.85rem;'>📄 Uploaded: {', '.join(st.session_state.uploaded_docs)}</p>",
        unsafe_allow_html=True
    )

# ===============================
# CHAT LOGIC (Smarter Adaptive RAG + WebSearch)
# ===============================
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    report_context = "\n\n".join(st.session_state.uploaded_texts) if st.session_state.uploaded_texts else None

    with st.spinner("🔍 Retrieving relevant information..."):
        try:
            context, sources = get_relevant_context(user_query, k=6, report_text=report_context)
            st.session_state.source_used = "RAG"
        except Exception as e:
            st.error(f"Retrieval Error: {e}")
            context, sources = "", []

    # Evaluate context quality
    context_length = len(context.split())
    medical_keywords = ["disease", "health", "treatment", "therapy", "symptom", "diagnosis", "patient", "doctor"]
    has_medical_relevance = any(word.lower() in context.lower() for word in medical_keywords)

    # Decide if we should fallback BEFORE calling the LLM
    if (context_length < 250 or not has_medical_relevance) and ENABLE_WEB_SEARCH:
        with st.spinner("🌐 Searching the web for better context..."):
            web_context = google_search(user_query)
            if web_context.strip():
                context += "\n\n" + web_context
                sources = list(set(sources + ["Google Search"]))
                st.session_state.source_used = "Web Search"

    # Generate answer
    if not context.strip():
        response = "⚠️ Sorry, I couldn’t find relevant information. Please rephrase your question."
    else:
        with st.spinner("🧠 Generating response..."):
            try:
                response = generate_answer(
                    user_query,
                    context,
                    response_mode=st.session_state.response_mode,
                    sources=sources
                )

                # Only if model explicitly says "not enough info" → do one more web fallback
                insufficient_phrases = [
                    "not enough medical information",
                    "insufficient information",
                    "unable to answer confidently",
                ]
                if any(p in response.lower() for p in insufficient_phrases) and ENABLE_WEB_SEARCH:
                    with st.spinner("🔄 Fetching additional web insights..."):
                        web_context = google_search(user_query)
                        if web_context.strip():
                            context += "\n\n" + web_context
                            sources = list(set(sources + ["Google Search"]))
                            st.session_state.source_used = "Web Search"
                            response = generate_answer(
                                user_query,
                                context,
                                response_mode=st.session_state.response_mode,
                                sources=sources
                            )

            except Exception as e:
                response = f"⚠️ Error generating answer: {e}"

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources if sources else ["General medical knowledge"]
    })

    st.caption(f"🧠 Source: {st.session_state.source_used} | References: {', '.join(sources)}")
    st.rerun()
