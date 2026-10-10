import streamlit as st
import io
import os
import pandas as pd
from extractor import DocumentExtractor
from vector_store import StudyMateVectorStore, ContextPreservingChunker

st.set_page_config(
    page_title="StudyMate AI - Document Extractor & Vector Store",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache vector store instance so model and ChromaDB client load once
@st.cache_resource
def load_vector_store():
    return StudyMateVectorStore.get_instance(persist_dir="./data/chroma_db", model_name="all-MiniLM-L6-v2")

# Automatic Vector Store Clean-up on Page Reload / Refresh
if "session_initialized" not in st.session_state:
    st.session_state["session_initialized"] = True
    st.session_state["active_file"] = None
    vs = load_vector_store()
    vs.clear_all()

# Sidebar Vector Store Management
with st.sidebar:
    st.markdown("### 🧹 Vector Store Control")
    st.caption("Vectors and indexes automatically clear on refresh or when uploading a new file.")
    if st.button("🗑️ Wipe & Reset Vector Store", use_container_width=True):
        vs = load_vector_store()
        vs.clear_all()
        st.session_state["active_file"] = None
        st.success("Vector store completely purged!")
        st.rerun()

# Custom Clean High-Contrast Styling
st.markdown("""
<style>
    .hero-header {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        background-color: #ede9fe;
        color: #5b21b6;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        display: inline-block;
    }
    .similarity-badge {
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        display: inline-block;
    }
    .page-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 0.8rem;
    }
    .chunk-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .image-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .notice-box {
        background-color: #f1f5f9;
        border-left: 4px solid #6366f1;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        color: #1e293b;
    }
    .breadcrumb-box {
        background-color: #eff6ff;
        border: 1px dashed #93c5fd;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        font-family: monospace;
        font-size: 0.85rem;
        color: #1e40af;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-header">
    <h1 style="margin: 0; color: #1e1b4b; font-size: 2.2rem;">📚 StudyMate AI — Deep Extractor & Vector Knowledge Base</h1>
    <p style="margin: 0.4rem 0 0 0; color: #475569; font-size: 1.05rem;">
        Extracts <strong>Text, Headings, Images/Diagrams, Tables</strong> and converts content into <strong>Context-Preserved Dense Embeddings</strong> stored in <strong>ChromaDB</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# Sample material text for 1-click test
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

# Upload Section
upload_col, action_col = st.columns([3, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload Course Document or Image",
        type=["pdf", "docx", "pptx", "txt", "md", "png", "jpg", "jpeg", "webp"],
        help="Upload PDF textbook notes, Word documents, PowerPoint lecture slides, or standalone images"
    )

with action_col:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    load_sample = st.button("🧪 Test with Sample Notes", use_container_width=True)

# Determine data source
file_name = None
file_bytes = None

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
elif load_sample:
    file_name = "Computer_Networks_Chapter_3.txt"
    file_bytes = SAMPLE_TEXT.encode("utf-8")

if file_bytes is not None:
    # Auto-clear vector database when a different document is uploaded
    if st.session_state.get("active_file") != file_name:
        st.session_state["active_file"] = file_name
        vs = load_vector_store()
        vs.clear_all()

    with st.spinner(f"Extracting all text, images, tables, and metadata from '{file_name}'..."):
        result = DocumentExtractor.extract_from_bytes(file_name, file_bytes)

    # Safe references
    all_images = getattr(result, "all_images", [])
    all_tables = getattr(result, "all_tables", [])
    units = getattr(result, "units", [])
    headings = getattr(result, "headings", [])

    st.success(f"Deep extraction complete for **{result.filename}** ({result.file_type})!")

    # Metric Cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Format", result.file_type.split()[0])
    m2_label = "Slides" if "PowerPoint" in result.file_type else ("Pages" if "PDF" in result.file_type else "Sections")
    m2.metric(m2_label, result.total_units)
    m3.metric("Word Count", f"{result.total_words:,}")
    m4.metric("Headings", len(headings))
    m5.metric("Images Extracted", len(all_images))
    m6.metric("Tables Found", len(all_tables))

    st.markdown("---")

    # Content Display Tabs
    tab_outline, tab_viewer, tab_images, tab_tables, tab_vectors, tab_raw = st.tabs([
        "📑 Extracted Outline & Headings",
        f"📖 Page/Slide Viewer ({result.total_units})",
        f"🖼️ Extracted Images ({len(all_images)})",
        f"📊 Extracted Tables ({len(all_tables)})",
        "🧠 Vector Store & Embeddings",
        "📜 Complete Extracted Text"
    ])

    # Tab 1: Outline
    with tab_outline:
        st.subheader("📑 Detected Headings & Section Outline")
        if headings:
            st.caption("All chapter, unit, and section titles discovered in the document:")
            col_h1, col_h2 = st.columns(2)
            for idx, heading in enumerate(headings):
                target_col = col_h1 if idx % 2 == 0 else col_h2
                target_col.markdown(f"**{idx + 1}.** 🏷️ `{heading}`")
        else:
            st.info("ℹ️ No explicit headings detected; document content is structured by pages/sections.")

    # Tab 2: Page / Slide Viewer
    with tab_viewer:
        if units:
            unit_label = units[0].unit_type
            st.subheader(f"📖 Inspect Content by {unit_label}")

            unit_options = [f"{u.unit_type} {u.unit_number}: {u.heading}" for u in units]
            selected_idx = st.selectbox(
                f"Choose {unit_label}",
                range(len(unit_options)),
                format_func=lambda i: unit_options[i]
            )

            current_unit = units[selected_idx]

            unit_images = getattr(current_unit, "images", [])
            unit_tables = getattr(current_unit, "tables", [])

            st.markdown(f"""
            <div class="page-box">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.5rem; margin-bottom: 1rem;">
                    <div>
                        <span class="metric-badge">{current_unit.unit_type} {current_unit.unit_number}</span>
                        <strong style="margin-left: 0.5rem; font-size: 1.15rem; color: #1e293b;">{current_unit.heading}</strong>
                    </div>
                    <span style="color: #64748b; font-size: 0.9rem;">{current_unit.word_count} words • {len(unit_images)} image(s) • {len(unit_tables)} table(s)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Display speaker notes if present
            if getattr(current_unit, "speaker_notes", None):
                with st.expander("🎙️ Speaker Notes for this Slide", expanded=True):
                    st.info(current_unit.speaker_notes)

            # Display unit text
            st.markdown("##### Extracted Text Content:")
            st.text_area(
                "Text",
                value=current_unit.text,
                height=250,
                key=f"text_area_{selected_idx}"
            )

            # Display images on this page/slide
            if unit_images:
                st.markdown(f"##### 🖼️ Images / Diagrams on {current_unit.unit_type} {current_unit.unit_number}:")
                img_cols = st.columns(min(len(unit_images), 3))
                for idx, img in enumerate(unit_images):
                    col = img_cols[idx % len(img_cols)]
                    with col:
                        st.image(img.image_bytes, caption=img.caption, use_container_width=True)
                        st.download_button(
                            f"⬇️ Download Image",
                            data=img.image_bytes,
                            file_name=f"{img.image_id}.{img.image_format}",
                            mime=f"image/{img.image_format}",
                            key=f"btn_dl_{img.image_id}",
                            use_container_width=True
                        )

            # Display tables on this page/slide
            if unit_tables:
                st.markdown(f"##### 📊 Tables on {current_unit.unit_type} {current_unit.unit_number}:")
                for tbl in unit_tables:
                    st.caption(f"**{tbl.title}**")
                    if tbl.rows:
                        df = pd.DataFrame(tbl.rows, columns=tbl.headers if tbl.headers else None)
                        st.dataframe(df, use_container_width=True)

        else:
            st.warning("No readable units found in this document.")

    # Tab 3: All Images Gallery
    with tab_images:
        st.subheader("🖼️ All Extracted Images, Diagrams & Figures")
        if all_images:
            st.caption(f"Found **{len(all_images)}** embedded images across the document:")
            grid_cols = st.columns(3)
            for idx, img in enumerate(all_images):
                col = grid_cols[idx % 3]
                with col:
                    st.markdown(f"""
                    <div class="image-card">
                        <span class="metric-badge">{img.unit_type} {img.unit_number}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(img.image_bytes, caption=img.caption, use_container_width=True)
                    st.download_button(
                        f"⬇️ Download ({img.width}x{img.height})",
                        data=img.image_bytes,
                        file_name=f"{img.image_id}.{img.image_format}",
                        mime=f"image/{img.image_format}",
                        key=f"gallery_dl_{img.image_id}",
                        use_container_width=True
                    )
        else:
            st.markdown("""
            <div class="notice-box">
                <strong>ℹ️ No embedded images found in this document.</strong><br>
                All text content, headings, and document structure have been fully extracted into the other tabs.
            </div>
            """, unsafe_allow_html=True)

    # Tab 4: All Tables Gallery
    with tab_tables:
        st.subheader("📊 All Extracted Structured Tables")
        if all_tables:
            st.caption(f"Found **{len(all_tables)}** tables across the document:")
            for tbl in all_tables:
                st.markdown(f"**📌 {tbl.title}**")
                if tbl.rows:
                    df = pd.DataFrame(tbl.rows, columns=tbl.headers if tbl.headers else None)
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        f"⬇️ Download {tbl.title} as CSV",
                        data=csv,
                        file_name=f"{tbl.table_id}.csv",
                        mime="text/csv",
                        key=f"csv_dl_{tbl.table_id}"
                    )
                st.markdown("---")
        else:
            st.markdown("""
            <div class="notice-box">
                <strong>ℹ️ No structured tables detected in this document.</strong>
            </div>
            """, unsafe_allow_html=True)

    # Tab 5: Vector Store & Embeddings
    with tab_vectors:
        st.subheader("🧠 Context-Preserving Embeddings & Vector Database")
        st.caption(
            "Transforms document units into dense vector embeddings using **SentenceTransformers (`all-MiniLM-L6-v2`)** "
            "and stores them with full hierarchical context in **ChromaDB**."
        )

        vector_store = load_vector_store()

        cfg_col1, cfg_col2, cfg_col3 = st.columns([2, 2, 2])
        with cfg_col1:
            chunk_size = st.slider("Target Chunk Size (chars)", min_value=200, max_value=1200, value=500, step=50)
        with cfg_col2:
            chunk_overlap = st.slider("Sliding Overlap (chars)", min_value=30, max_value=300, value=100, step=10)
        with cfg_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            index_now = st.button("⚡ Generate Embeddings & Index in ChromaDB", type="primary", use_container_width=True)

        # Trigger Indexing
        if index_now:
            with st.spinner("Generating context-rich embeddings and writing to ChromaDB..."):
                index_result = vector_store.index_document(
                    doc_result=result,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            st.success(
                f"Successfully embedded and indexed **{index_result['chunks_indexed']} chunks** "
                f"for `{index_result['filename']}` ({index_result['dimensions']} dimensions, model: `{index_result['model_name']}`)!"
            )

        # Fetch stored chunks for this document
        stored_chunks = vector_store.get_document_chunks(result.filename)

        st.markdown("---")

        # Two-column layout: Search Tester & Chunk Inspector
        vcol1, vcol2 = st.columns([1, 1])

        with vcol1:
            st.markdown("#### 🔍 Advanced Context Retrieval Engine")
            st.caption("Combines **Dense Embeddings + BM25 Keywords (RRF)** with **Surrounding Context Window Expansion**:")

            search_query = st.text_input(
                "Enter test search query",
                placeholder="e.g. How does 3-way handshake work? Or: SYN-ACK sequence number",
                key="vector_search_input"
            )

            # Retrieval Strategy & Expansion Controls
            mode_col, topk_col = st.columns([2, 1])
            with mode_col:
                retrieval_mode = st.selectbox(
                    "Retrieval Strategy",
                    ["🚀 Hybrid Search (Dense + BM25 RRF)", "🧠 Pure Dense Vector (Cosine)", "🔤 Pure BM25 Keyword Search"],
                    index=0
                )
            with topk_col:
                top_k_val = st.slider("Max Results (Top K)", min_value=1, max_value=6, value=3)

            # Context Window Expansion Options
            exp_col1, exp_col2 = st.columns([3, 2])
            with exp_col1:
                expand_ctx = st.checkbox("🧩 Expand Context Window (Stitch Neighbors)", value=True, help="Automatically stitches previous and next chunks so surrounding explanations are never cut off")
            with exp_col2:
                if "Hybrid" in retrieval_mode:
                    alpha_val = st.slider("Dense / Lexical Balance", min_value=0.1, max_value=0.9, value=0.6, step=0.1, help="0.6 = 60% semantic vectors, 40% exact keyword matches")
                else:
                    alpha_val = 0.6

            if search_query.strip():
                if "Hybrid" in retrieval_mode:
                    matches = vector_store.hybrid_search(
                        query=search_query,
                        top_k=top_k_val,
                        filename_filter=result.filename,
                        alpha=alpha_val,
                        expand_context=expand_ctx,
                        window_size=1
                    )
                elif "Dense" in retrieval_mode:
                    matches = vector_store.search_dense(
                        query=search_query,
                        top_k=top_k_val,
                        filename_filter=result.filename
                    )
                    # Add expansion manually if requested
                    if expand_ctx:
                        for m in matches:
                            exp_d = vector_store.expand_chunk_context(m["chunk_id"], window_size=1)
                            m["expanded_text"] = exp_d["expanded_text"]
                            m["has_expanded"] = exp_d["has_expanded"]
                else:
                    matches = vector_store.search_bm25(
                        query=search_query,
                        top_k=top_k_val,
                        filename_filter=result.filename
                    )
                    if expand_ctx:
                        for m in matches:
                            exp_d = vector_store.expand_chunk_context(m["chunk_id"], window_size=1)
                            m["expanded_text"] = exp_d["expanded_text"]
                            m["has_expanded"] = exp_d["has_expanded"]

                if matches:
                    st.markdown(f"**Found {len(matches)} relevant match(es):**")
                    for m in matches:
                        meta = m["metadata"]
                        sim_pct = m.get("hybrid_score", round(m.get("score", 0.0) * 100, 1))
                        heading_ref = meta.get("heading", "Topic")
                        unit_num = meta.get("unit_number", 1)
                        unit_type = meta.get("unit_type", "Unit")
                        d_rank = m.get("dense_rank", "-")
                        b_rank = m.get("bm25_rank", "-")

                        rank_badges = ""
                        if d_rank != "-":
                            rank_badges += f"<span style='background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:0.75rem; margin-right:4px;'>Vector: #{d_rank}</span>"
                        if b_rank != "-":
                            rank_badges += f"<span style='background:#fef3c7; color:#92400e; padding:2px 6px; border-radius:4px; font-size:0.75rem;'>BM25: #{b_rank}</span>"

                        st.markdown(f"""
                        <div class="chunk-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <div>
                                    <span class="metric-badge">{unit_type} {unit_num}</span>
                                    <strong style="margin-left: 0.5rem; color: #1e293b;">{heading_ref}</strong>
                                </div>
                                <div>
                                    {rank_badges}
                                    <span class="similarity-badge">🎯 {sim_pct}% Confidence</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Render expanded contiguous context if active
                        if m.get("has_expanded") and m.get("expanded_text"):
                            st.caption("📖 Contiguous Narrative with Surrounding Context:")
                            st.text_area(
                                "Expanded Context Passage",
                                value=m["expanded_text"],
                                height=160,
                                key=f"exp_{m['chunk_id']}"
                            )
                        else:
                            st.markdown(f"""
                            <div style="color: #334155; font-size: 0.95rem; line-height: 1.5; background: #f8fafc; padding: 0.8rem; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 0.6rem;">
                                {m['text']}
                            </div>
                            """, unsafe_allow_html=True)

                        # Show associated images if chunk has them
                        if meta.get("has_images") and all_images:
                            matching_imgs = [img for img in all_images if img.unit_number == unit_num]
                            if matching_imgs:
                                with st.expander(f"🖼️ View {len(matching_imgs)} Diagram(s) associated with {unit_type} {unit_num}"):
                                    img_sub_cols = st.columns(min(len(matching_imgs), 3))
                                    for idx, img in enumerate(matching_imgs):
                                        col = img_sub_cols[idx % len(img_sub_cols)]
                                        col.image(img.image_bytes, caption=img.caption, use_container_width=True)

                else:
                    st.info("No matches found. Make sure to click **'⚡ Generate Embeddings & Index in ChromaDB'** first!")

        with vcol2:
            st.markdown("#### 📦 Context Integrity & Chunks Inspector")
            if stored_chunks:
                st.caption(f"Currently storing **{len(stored_chunks)} context-preserved chunks** in ChromaDB:")
                for chk in stored_chunks:
                    meta = chk["metadata"]
                    c_idx = meta.get("chunk_index", 1)
                    c_tot = meta.get("total_chunks", len(stored_chunks))
                    heading = meta.get("heading", "Topic")
                    u_type = meta.get("unit_type", "Unit")
                    u_num = meta.get("unit_number", 1)
                    w_count = meta.get("word_count", 0)

                    with st.expander(f"Chunk #{c_idx}/{c_tot} • {u_type} {u_num}: {heading} ({w_count} words)"):
                        st.markdown(f"""
                        <div class="breadcrumb-box">
                            📌 Breadcrumb Context Vector: [Source: {meta.get('filename')} | {u_type} {u_num} | Topic: {heading}]
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("**Raw Chunk Content:**")
                        st.code(chk["text"], language="markdown")
            else:
                st.info(
                    "No chunks currently in vector database for this document. "
                    "Click **'⚡ Generate Embeddings & Index in ChromaDB'** to generate dense vectors!"
                )

        # Database Stats Section
        st.markdown("---")
        stat_h_col, stat_btn_col = st.columns([3, 1])
        with stat_h_col:
            st.markdown("##### 🗄️ ChromaDB Collection Statistics")
        with stat_btn_col:
            if st.button("🗑️ Wipe Stored Vectors", key="wipe_vectors_btn", use_container_width=True):
                vector_store.clear_all()
                st.success("Vector store completely cleared!")
                st.rerun()

        stats = vector_store.get_stats()
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Stored Vectors", stats["total_chunks"])
        sc2.metric("Indexed Documents", stats["total_documents"])
        sc3.metric("Vector Dimensions", stats["dimension"])
        sc4.metric("Storage Path", os.path.basename(stats["persist_directory"]))

    # Tab 6: Complete Text & Search
    with tab_raw:
        st.subheader("📜 Complete Extracted Content")
        search_query_raw = st.text_input("🔍 Search within extracted text", placeholder="e.g. Handshake, SYN, protocol...")

        all_text = "\n\n".join([f"--- [{u.unit_type} {u.unit_number}: {u.heading}] ---\n{u.text}" for u in units])

        if search_query_raw.strip():
            matches = [u for u in units if search_query_raw.lower() in u.text.lower()]
            unit_kind = units[0].unit_type.lower() if units else "section"
            st.info(f"Found {len(matches)} matching {unit_kind}(s) for **'{search_query_raw}'**:")
            for m in matches:
                with st.expander(f"📍 {m.unit_type} {m.unit_number}: {m.heading}"):
                    st.write(m.text)
        else:
            st.text_area("Full Document Text", value=all_text, height=400)

        st.download_button(
            "💾 Download All Extracted Text (.txt)",
            data=all_text,
            file_name=f"extracted_{result.filename}.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    st.info("👆 Please upload a **PDF**, **PPTX**, **DOCX**, **TXT**, or **Image** above, or click **'Test with Sample Notes'**!")
