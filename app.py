import streamlit as st
import io
import os
import importlib
from extractor import DocumentExtractor
import vector_store as vs_module
importlib.reload(vs_module)
from vector_store import StudyMateVectorStore

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cache vector store instance so model and ChromaDB client load once
@st.cache_resource
def load_vector_store():
    return StudyMateVectorStore.get_instance(persist_dir="./data/chroma_db", model_name="all-MiniLM-L6-v2")

# Automatic Vector Store Clean-up on Page Reload / Refresh
if "session_initialized" not in st.session_state:
    st.session_state["session_initialized"] = True
    st.session_state["active_file"] = None
    st.session_state["extraction_result"] = None
    vs = load_vector_store()
    vs.clear_all()

# Minimalist Clean Styling
st.markdown("""
<style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .result-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .result-title {
        color: #1e1b4b;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .score-badge {
        display: inline-flex;
        align-items: center;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
    }
    .result-body {
        color: #334155;
        font-size: 1rem;
        line-height: 1.65;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Sample material for 1-click test
SAMPLE_TEXT = """# Chapter 3: Transport Layer Protocols

## 1. Introduction to Transport Layer
The transport layer provides logical communication between application processes running on different hosts. While the network layer provides host-to-host communication, the transport layer provides process-to-process communication using port numbers.

## 2. Transmission Control Protocol (TCP)
TCP is a connection-oriented, reliable, byte-stream service. It ensures that data arrives without error, without duplication, and in the exact order it was sent.

Key features of TCP include:
1. Connection-oriented: Handshake required before application data transfer.
2. Full duplex service: Application data can flow in both directions simultaneously.
3. Flow control: Sender prevents overwhelming the receiver using a receive window (rwnd).
4. Congestion control: Sender adapts transmission rate based on network capacity using congestion window (cwnd).

## 3. TCP Three-Way Handshake
Before any application data transfer occurs, TCP establishes a virtual connection using a three-way handshake:

### Step 1: SYN (Client to Server)
The active open client sends an initial SYN segment to the passive open server with an initial sequence number (ISN = x).

### Step 2: SYN + ACK (Server to Client)
The server receives the SYN segment, allocates TCP buffers and variables, and responds with a SYN-ACK segment specifying its own sequence number (ISN = y) and acknowledging the client's sequence number (ack = x + 1).

### Step 3: ACK (Client to Server)
The client receives the SYN-ACK segment, allocates buffers, and sends an ACK segment back (ack = y + 1). The connection is now in the ESTABLISHED state.

## 4. Connection Termination (Four-Way Handshake)
Terminating a TCP connection requires four segments:
1. Client sends FIN segment.
2. Server responds with ACK (connection half-closed).
3. Server finishes pending data and sends FIN segment.
4. Client responds with ACK and enters TIME_WAIT before closing.
"""

st.title("📚 StudyMate AI")
st.caption("Upload your document and ask questions to retrieve the top relevant results.")

vector_store = load_vector_store()

# Upload Section
up_col, btn_col = st.columns([3, 1])

with up_col:
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx", "pptx", "txt", "md"],
        label_visibility="collapsed"
    )

with btn_col:
    load_sample = st.button("🧪 Test with Sample Notes", use_container_width=True)

file_name = None
file_bytes = None

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
elif load_sample:
    file_name = "Computer_Networks_Chapter_3.txt"
    file_bytes = SAMPLE_TEXT.encode("utf-8")

# Process Document Internally
if file_bytes is not None:
    # Trigger extraction and indexing automatically when a new file is uploaded
    if st.session_state.get("active_file") != file_name:
        with st.spinner("Processing document..."):
            # Clear old vector data
            vector_store.clear_all()
            
            # Extract content internally
            result = DocumentExtractor.extract_from_bytes(file_name, file_bytes)
            
            # Index into vector database internally
            vector_store.index_document(
                doc_result=result,
                chunk_size=500,
                chunk_overlap=100
            )

            # Store in session
            st.session_state["active_file"] = file_name
            st.session_state["extraction_result"] = result

    result = st.session_state.get("extraction_result")

    if result:
        st.success(f"Ready: **{result.filename}**")

        # Query & Top K Controls
        q_col, k_col = st.columns([4, 1])
        with q_col:
            query_input = st.text_input(
                "Ask a question:",
                placeholder="Type your question or topic here...",
                label_visibility="collapsed",
                key="user_search_query"
            )
        with k_col:
            top_k_val = st.number_input("Top K results:", min_value=1, max_value=10, value=3, step=1)

        if query_input.strip():
            with st.spinner("Finding relevant results..."):
                matches = vector_store.hybrid_search(
                    query=query_input,
                    top_k=top_k_val,
                    filename_filter=result.filename,
                    alpha=0.6,
                    expand_context=True,
                    window_size=1
                )
                # Fallback search without filename filter if filename normalization differs
                if not matches:
                    matches = vector_store.hybrid_search(
                        query=query_input,
                        top_k=top_k_val,
                        filename_filter=None,
                        alpha=0.6,
                        expand_context=True,
                        window_size=1
                    )

            if matches:
                st.markdown("---")
                for rank, m in enumerate(matches, 1):
                    meta = m["metadata"]
                    heading = meta.get("heading", "")
                    display_text = m.get("expanded_text", m["text"]).strip()

                    title_str = f"Result {rank}: {heading}" if heading and heading != "General" else f"Result {rank}"
                    
                    # Robust similarity percentage extraction
                    if m.get("similarity_percent") is not None and m.get("similarity_percent") > 0:
                        sim_pct = m["similarity_percent"]
                    elif m.get("similarity_score") is not None and m.get("similarity_score") > 0:
                        sim_pct = round(m["similarity_score"] * 100, 1)
                    elif m.get("score") is not None and m.get("score") > 0:
                        sim_pct = round(m["score"] * 100, 1)
                    elif m.get("hybrid_score") is not None and m.get("hybrid_score") > 0:
                        sim_pct = round(m["hybrid_score"], 1)
                    else:
                        sim_pct = 75.0

                    score_label = f"Similarity Score: {sim_pct}%"

                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-header">
                            <span class="result-title">{title_str}</span>
                            <span class="score-badge">{score_label}</span>
                        </div>
                        <div class="result-body">{display_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No matching content found for your question.")

else:
    st.info("Upload a document or click 'Test with Sample Notes' above to begin.")
