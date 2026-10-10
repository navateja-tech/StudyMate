import streamlit as st
import io
import pandas as pd
from extractor import DocumentExtractor

st.set_page_config(
    page_title="StudyMate AI - Deep Content & Media Extractor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Clean Styling
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
    .page-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 0.8rem;
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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-header">
    <h1 style="margin: 0; color: #1e1b4b; font-size: 2.2rem;">📚 StudyMate AI — Deep Content & Media Extractor</h1>
    <p style="margin: 0.4rem 0 0 0; color: #475569; font-size: 1.05rem;">
        Extracts <strong>all Text, Headings, Embedded Diagrams/Images, Tables, and Slide Notes</strong> from PDFs, PPTX slides, Word documents, and images.
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
    tab_outline, tab_viewer, tab_images, tab_tables, tab_raw = st.tabs([
        "📑 Extracted Outline & Headings",
        f"📖 Page/Slide Viewer ({result.total_units})",
        f"🖼️ Extracted Images ({len(all_images)})",
        f"📊 Extracted Tables ({len(all_tables)})",
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

    # Tab 5: Complete Text & Search
    with tab_raw:
        st.subheader("📜 Complete Extracted Content")
        search_query = st.text_input("🔍 Search within extracted text", placeholder="e.g. Handshake, SYN, protocol...")

        all_text = "\n\n".join([f"--- [{u.unit_type} {u.unit_number}: {u.heading}] ---\n{u.text}" for u in units])

        if search_query.strip():
            matches = [u for u in units if search_query.lower() in u.text.lower()]
            unit_kind = units[0].unit_type.lower() if units else "section"
            st.info(f"Found {len(matches)} matching {unit_kind}(s) for **'{search_query}'**:")
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
