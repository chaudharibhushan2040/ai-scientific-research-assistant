import streamlit as st
import os
import tiktoken
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from core.document_loader import load_and_split_multiple_pdfs
from core.vector_store import create_vector_store
from core.llm import get_llm

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="AI Scientific Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# ----------------- DARK THEME -----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ----------------- ANIMATED HEADER -----------------
st.markdown("""
<h1 style='text-align: center; color: cyan;'>
🧠 AI Scientific Research Assistant
</h1>
<hr style='border:1px solid #00FFFF'>
""", unsafe_allow_html=True)

st.markdown("### Upload documents and perform intelligent scientific analysis.")

# ----------------- SESSION STATE -----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm" not in st.session_state:
    st.session_state.llm = get_llm()

# ----------------- FILE UPLOAD -----------------
uploaded_files = st.file_uploader(
    "Upload Multiple PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    file_paths = []

    for uploaded_file in uploaded_files:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        file_paths.append(file_path)

    with st.spinner("🔬 Processing documents..."):
        chunks = load_and_split_multiple_pdfs(file_paths)
        st.session_state.vectorstore = create_vector_store(chunks)

    st.success(f"✅ {len(uploaded_files)} document(s) processed successfully!")

    # File preview
    st.markdown("### 📂 Uploaded Files:")
    for file in uploaded_files:
        st.write(f"• {file.name}")

# ----------------- AUTO SUMMARY -----------------
if st.session_state.vectorstore:
    if st.button("📄 Generate Document Summary"):
        with st.spinner("Generating summary..."):
            docs = st.session_state.vectorstore.similarity_search("Summarize the document", k=5)
            context = "\n\n".join([doc.page_content for doc in docs])

            summary_prompt = f"""
            Provide a professional executive summary of the following content:

            {context}
            """

            summary = st.session_state.llm.invoke(summary_prompt)
            st.subheader("📑 Executive Summary")
            st.write(summary.content)

# ----------------- CHAT SECTION -----------------
if st.session_state.vectorstore:

    query = st.chat_input("Ask a research question...")

    if query:

        with st.spinner("🧠 Analyzing..."):
            retrieved_docs = st.session_state.vectorstore.similarity_search(query, k=3)

            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            prompt = f"""
            You are an advanced scientific AI assistant.
            Answer strictly using the provided context.
            Provide clear and professional explanation.

            Context:
            {context}

            Question:
            {query}

            Answer:
            """

            response = st.session_state.llm.invoke(prompt)

        # Token estimation
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(prompt + response.content))

        st.session_state.chat_history.append((query, response.content, token_count, retrieved_docs))

    # Display chat
    for q, a, tokens, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)

        with st.chat_message("assistant"):
            st.write(a)
            st.caption(f"🔢 Estimated Tokens Used: {tokens}")

            with st.expander("📚 Source References"):
                for doc in sources:
                    st.write(doc.page_content[:300] + "...")

            # Download as PDF
            if st.button("⬇ Download Answer as PDF", key=q):
                pdf_path = "answer.pdf"
                doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                styles = getSampleStyleSheet()
                elements = [Paragraph(a, styles["Normal"])]
                doc.build(elements)

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name="AI_Answer.pdf",
                        mime="application/pdf"
                    )