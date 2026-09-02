"""
app.py

Streamlit front-end for the Research Paper Intelligence & Lightweight RAG System.

Run with:
    streamlit run app.py
"""

import streamlit as st

from src import config
from src import document_manager
from src import rag

st.set_page_config(
    page_title="Research Paper Intelligence",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "processed_files" not in st.session_state:
    # Tracks filenames already processed in this session, to avoid re-embedding
    # a freshly-uploaded (unchanged) file on every Streamlit rerun.
    st.session_state.processed_files = set()

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_summary" not in st.session_state:
    st.session_state.last_summary = None


def refresh_active_documents():
    st.session_state.active_documents = document_manager.get_active_documents()


if "active_documents" not in st.session_state:
    refresh_active_documents()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📚 Research Paper Intelligence")
st.caption(
    "Upload research papers and ask natural-language questions. "
    "Answers are generated **only** from the content of your uploaded papers — "
    "never from outside knowledge."
)

if not config.is_groq_configured():
    st.warning(
        "⚠️ GROQ_API_KEY is not configured. Add it to a `.env` file "
        "(see `.env.example`) before asking questions. You can still upload "
        "and process papers without it.",
        icon="⚠️",
    )

st.divider()

# ---------------------------------------------------------------------------
# Sidebar: upload & manage papers
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📤 Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload one or more research paper PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Both digitally-generated and scanned PDFs are supported.",
    )

    process_clicked = st.button("Process Papers", type="primary", use_container_width=True)

    if process_clicked:
        if not uploaded_files:
            st.error("Please choose at least one PDF file first.")
        else:
            progress = st.progress(0.0, text="Starting...")
            errors = []
            for i, uploaded_file in enumerate(uploaded_files):
                progress.progress(
                    i / len(uploaded_files),
                    text=f"Processing {uploaded_file.name}...",
                )
                try:
                    file_bytes = uploaded_file.getvalue()
                    result = document_manager.process_uploaded_pdf(
                        uploaded_file.name, file_bytes
                    )
                    st.session_state.processed_files.add(uploaded_file.name)
                    msg = (
                        f"✅ {result.filename}: {result.page_count} pages, "
                        f"{result.chunk_count} chunks indexed"
                    )
                    if result.ocr_pages_used:
                        msg += f" ({result.ocr_pages_used} page(s) used OCR)"
                    st.success(msg)
                except document_manager.IngestionError as exc:
                    errors.append(str(exc))
                except Exception as exc:
                    errors.append(f"Unexpected error processing '{uploaded_file.name}': {exc}")

            progress.progress(1.0, text="Done")
            for err in errors:
                st.error(f"❌ {err}")

            refresh_active_documents()

    st.divider()
    st.header("📑 Active Papers")

    active_docs = st.session_state.active_documents
    if not active_docs:
        st.info("No papers indexed yet. Upload and process PDFs above.")
        selected_doc_ids = []
    else:
        st.caption("Select which papers to include when asking a question:")
        selected_doc_ids = []
        for doc in active_docs:
            checked = st.checkbox(
                doc["filename"], value=True, key=f"select_{doc['doc_id']}"
            )
            if checked:
                selected_doc_ids.append(doc["doc_id"])

        st.divider()
        if st.button("🗑️ Clear All Papers", use_container_width=True):
            for doc in active_docs:
                document_manager.remove_document(doc["doc_id"])
            st.session_state.processed_files = set()
            st.session_state.last_answer = None
            st.session_state.last_summary = None
            refresh_active_documents()
            st.rerun()

# ---------------------------------------------------------------------------
# Main area: question answering
# ---------------------------------------------------------------------------
tab_ask, tab_summarize = st.tabs(["❓ Ask a Question", "🧾 Summarize a Paper"])

with tab_ask:
    st.subheader("Ask a question about your selected papers")

    example_questions = [
        "What is the main contribution of this paper?",
        "What dataset was used?",
        "What methodology was proposed?",
        "What are the limitations?",
        "Compare the methodologies used in these papers.",
    ]
    st.caption("Examples: " + " · ".join(f"_{q}_" for q in example_questions))

    question = st.text_area("Your question", height=80, key="question_input")
    ask_clicked = st.button("Ask Question", type="primary")

    if ask_clicked:
        active_docs = st.session_state.active_documents
        if not active_docs:
            st.error("No papers have been uploaded and processed yet.")
        elif not selected_doc_ids:
            st.error("Please select at least one active paper to search.")
        elif not question or not question.strip():
            st.error("Please type a question first.")
        else:
            with st.spinner("Retrieving relevant passages and generating a grounded answer..."):
                try:
                    result = rag.answer_question(question, doc_ids=selected_doc_ids)
                    st.session_state.last_answer = result
                except rag.RAGError as exc:
                    st.error(str(exc))
                    st.session_state.last_answer = None
                except Exception as exc:
                    st.error(f"An unexpected error occurred: {exc}")
                    st.session_state.last_answer = None

    if st.session_state.last_answer:
        result = st.session_state.last_answer
        st.markdown("### 🧠 Answer")
        st.write(result["answer"])

        if result["citations"]:
            st.markdown("### 📌 Sources")
            for citation in result["citations"]:
                st.markdown(f"- {citation}")

with tab_summarize:
    st.subheader("Summarize a single paper")

    active_docs = st.session_state.active_documents
    if not active_docs:
        st.info("Upload and process a paper first.")
    else:
        options = {doc["filename"]: doc["doc_id"] for doc in active_docs}
        chosen_filename = st.selectbox("Choose a paper to summarize", list(options.keys()))
        summarize_clicked = st.button("Summarize Paper", type="primary")

        if summarize_clicked:
            with st.spinner(f"Summarizing '{chosen_filename}'..."):
                try:
                    doc_id = options[chosen_filename]
                    result = rag.summarize_paper(doc_id, chosen_filename)
                    st.session_state.last_summary = result
                except rag.RAGError as exc:
                    st.error(str(exc))
                    st.session_state.last_summary = None
                except Exception as exc:
                    st.error(f"An unexpected error occurred: {exc}")
                    st.session_state.last_summary = None

        if st.session_state.last_summary:
            summary_result = st.session_state.last_summary
            st.markdown("### 📄 Summary")
            st.write(summary_result["summary"])
            if summary_result["citations"]:
                st.markdown("### 📌 Sources")
                for citation in summary_result["citations"]:
                    st.markdown(f"- {citation}")

st.divider()
st.caption(
    "Research Paper Intelligence — a lightweight, locally-run RAG system. "
    "Answers are grounded strictly in the uploaded PDFs; nothing is invented."
)
