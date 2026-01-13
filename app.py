import os
import streamlit as st

from ingest import extract_text
from retriever import build_faiss, build_bm25
from query_expansion import expand_query
from llm import load_llm
from retriever import hybrid_search


# -------------------------------
# Ensure upload directory exists
# -------------------------------
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📄 Research Paper Chatbot")

#api_key = st.text_input("Enter DeepSeek API Key", type="password")
pdf = st.file_uploader("Upload Research Paper", type="pdf")

# -------------------------------
# Process PDF
# -------------------------------
if pdf:
    file_path = os.path.join(UPLOAD_DIR, pdf.name)

    with open(file_path, "wb") as f:
        f.write(pdf.read())

    text = extract_text(file_path)
    chunks = [c for c in text.split("\n\n") if len(c.strip()) > 50]

    faiss_index, _ = build_faiss(chunks)
    bm25 = build_bm25(chunks)

    query = st.text_input("Ask your question")

    if query:
        llm = load_llm()
        expanded_query = expand_query(llm, query)
        

        retrieved_chunks = hybrid_search(
        expanded_query,
        chunks,
        faiss_index,
        bm25
    )

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a research assistant.

Answer clearly and simply.
If math is present, explain it step by step.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    st.subheader("Answer")
    st.write(response.content)




