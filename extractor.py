import os
import io
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set
from PIL import Image
import pymupdf as fitz
from docx import Document as DocxDocument
from pptx import Presentation

class ExtractedImage:
    """Represents an image extracted from a document or presentation."""
    def __init__(
        self,
        image_id: str,
        unit_number: int,
        unit_type: str,
        image_bytes: bytes,
        image_format: str,
        width: int,
        height: int,
        caption: Optional[str] = None
    ):
        self.image_id = image_id
        self.unit_number = unit_number
        self.unit_type = unit_type
        self.image_bytes = image_bytes
        self.image_format = (image_format or "png").lower().replace("jpeg", "jpg")
        self.width = width
        self.height = height
        self.caption = caption or f"Figure from {unit_type} {unit_number}"

class ExtractedTable:
    """Represents a structured table extracted from a document or presentation."""
    def __init__(
        self,
        table_id: str,
        unit_number: int,
        unit_type: str,
        headers: List[str],
        rows: List[List[str]],
        title: Optional[str] = None
    ):
        self.table_id = table_id
        self.unit_number = unit_number
        self.unit_type = unit_type
        self.headers = headers or []
        self.rows = rows or []
        self.title = title or f"Table on {unit_type} {unit_number}"

class DocumentUnit:
    """Represents a single logical unit (e.g. Page, Slide, or Section)."""
    def __init__(
        self,
        unit_number: int,
        unit_type: str,  # "Page", "Slide", or "Section"
        text: str,
        heading: Optional[str] = None,
        images: Optional[List[ExtractedImage]] = None,
        tables: Optional[List[ExtractedTable]] = None,
        speaker_notes: Optional[str] = None
    ):
        self.unit_number = unit_number
        self.unit_type = unit_type
        self.text = (text or "").strip()
        self.heading = heading or f"{unit_type} {unit_number}"
        self.images = images if images is not None else []
        self.tables = tables if tables is not None else []
        self.speaker_notes = speaker_notes
        self.word_count = len(self.text.split()) if self.text else 0

class DocumentExtractionResult:
    """Consolidated document extraction result containing units, outline, and media."""
    def __init__(
        self,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        units: List[DocumentUnit],
        headings: List[str],
        all_images: Optional[List[ExtractedImage]] = None,
        all_tables: Optional[List[ExtractedTable]] = None
    ):
        self.filename = filename
        self.file_type = file_type
        self.file_size_bytes = file_size_bytes
        self.units = units if units is not None else []
        self.headings = headings if headings is not None else []
        self.total_units = len(self.units)
        self.total_words = sum(getattr(u, 'word_count', 0) for u in self.units)

        # Aggregate all images strictly from units
        if all_images is not None:
            self.all_images: List[ExtractedImage] = all_images
        else:
            self.all_images: List[ExtractedImage] = []
            for u in self.units:
                if hasattr(u, "images") and u.images:
                    self.all_images.extend(u.images)

        # Aggregate all tables strictly from units
        if all_tables is not None:
            self.all_tables: List[ExtractedTable] = all_tables
        else:
            self.all_tables: List[ExtractedTable] = []
            for u in self.units:
                if hasattr(u, "tables") and u.tables:
                    self.all_tables.extend(u.tables)

    def __getattr__(self, name: str):
        """Defensive attribute fallback."""
        if name == "all_images":
            imgs = []
            for u in getattr(self, "units", []):
                if hasattr(u, "images") and u.images:
                    imgs.extend(u.images)
            self.all_images = imgs
            return imgs
        elif name == "all_tables":
            tbls = []
            for u in getattr(self, "units", []):
                if hasattr(u, "tables") and u.tables:
                    tbls.extend(u.tables)
            self.all_tables = tbls
            return tbls
        raise AttributeError(f"'DocumentExtractionResult' object has no attribute '{name}'")


class DocumentExtractor:
    """
    Multi-format deep extractor for PDF, PPTX, DOCX, TXT, and Images.
    Strictly extracts ONLY actual content figures and tables belonging to the document,
    filtering out background templates, bullet icons, header logos, and package artifacts.
    """

    @staticmethod
    def _is_valid_content_image(width: int, height: int) -> bool:
        """
        Filters out decorative icons, bullet graphics, horizontal rule lines,
        and tiny spacers that are not genuine course diagrams/figures.
        """
        # Minimum resolution: at least 100x100 or total area >= 15,000 px
        if width < 100 or height < 100 or (width * height < 15000):
            return False

        # Extreme aspect ratio check (filters out 1200x5 divider lines or 5x800 border bars)
        aspect = width / max(height, 1)
        if aspect > 7.0 or aspect < 0.14:
            return False

        return True

    @staticmethod
    def extract_from_bytes(filename: str, file_bytes: bytes) -> DocumentExtractionResult:
        ext = os.path.splitext(filename)[1].lower()
        file_size = len(file_bytes)

        if ext == ".pdf":
            return DocumentExtractor._extract_pdf(filename, file_bytes, file_size)
        elif ext in [".docx", ".doc"]:
            return DocumentExtractor._extract_docx(filename, file_bytes, file_size)
        elif ext in [".pptx", ".ppt"]:
            return DocumentExtractor._extract_pptx(filename, file_bytes, file_size)
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            return DocumentExtractor._extract_standalone_image(filename, file_bytes, file_size)
        else:
            return DocumentExtractor._extract_text(filename, file_bytes, file_size)

    @staticmethod
    def _extract_pdf(filename: str, file_bytes: bytes, file_size: int) -> DocumentExtractionResult:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        units: List[DocumentUnit] = []
        headings: List[str] = []
        seen_headings = set()

        # Pass 1: Discover images and detect repeating template/header logos across pages
        # If an identical image hash appears on > 2 pages in a document, it's a template header/footer logo
        image_page_occurrences: Dict[str, Set[int]] = {}
        for page_idx in range(len(doc)):
            p = doc[page_idx]
            for img_info in p.get_images(full=True):
                try:
                    xref = img_info[0]
                    b_img = doc.extract_image(xref)
                    if b_img and b_img.get("image"):
                        h = hashlib.sha256(b_img["image"]).hexdigest()
                        if h not in image_page_occurrences:
                            image_page_occurrences[h] = set()
                        image_page_occurrences[h].add(page_idx + 1)
                except Exception:
                    continue

        repeating_logo_hashes = {
            h for h, pages in image_page_occurrences.items()
            if len(pages) > 2 and len(doc) >= 3
        }

        img_counter = 1

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1. Text extraction
            text = page.get_text("text").strip()

            # 2. Heading discovery
            page_heading = None
            try:
                blocks = page.get_text("blocks")
                for b in blocks:
                    block_text = b[4].strip()
                    if not block_text:
                        continue
                    first_line = block_text.split("\n")[0].strip()
                    if (re.match(r'^(chapter|unit|section|part|\d+(\.\d+)*)\b', first_line, re.IGNORECASE) or
                        (len(first_line) < 70 and (first_line.isupper() or first_line.istitle())) or
                        first_line.startswith("#")):
                        clean_h = first_line.lstrip("#").strip()
                        page_heading = clean_h
                        if clean_h not in seen_headings:
                            headings.append(clean_h)
                            seen_headings.add(clean_h)
                        break
            except Exception:
                pass

            # 3. Embedded Images extraction strictly for this page
            page_images: List[ExtractedImage] = []
            seen_page_xrefs = set()
            try:
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    if xref in seen_page_xrefs:
                        continue
                    seen_page_xrefs.add(xref)

                    try:
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            continue
                        img_bytes = base_image.get("image")
                        img_ext = base_image.get("ext", "png")
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)

                        if not img_bytes:
                            continue

                        # Check if it's a repeating header/footer logo
                        img_hash = hashlib.sha256(img_bytes).hexdigest()
                        if img_hash in repeating_logo_hashes:
                            continue

                        # Verify resolution and content validity
                        if DocumentExtractor._is_valid_content_image(width, height):
                            page_images.append(ExtractedImage(
                                image_id=f"pdf_img_{img_counter}",
                                unit_number=page_num,
                                unit_type="Page",
                                image_bytes=img_bytes,
                                image_format=img_ext,
                                width=width,
                                height=height,
                                caption=f"Figure {img_counter} (Page {page_num}, {width}x{height} px)"
                            ))
                            img_counter += 1
                    except Exception:
                        continue
            except Exception:
                pass

            # 4. Tables extraction
            page_tables: List[ExtractedTable] = []
            try:
                tables = page.find_tables()
                for t_idx, table in enumerate(tables):
                    table_data = table.extract()
                    if table_data and len(table_data) > 1:
                        headers = [str(col).strip() if col else f"Col {c_idx+1}" for c_idx, col in enumerate(table_data[0])]
                        rows = [[str(cell).strip() if cell else "" for cell in row] for row in table_data[1:]]
                        page_tables.append(ExtractedTable(
                            table_id=f"pdf_tbl_{page_num}_{t_idx+1}",
                            unit_number=page_num,
                            unit_type="Page",
                            headers=headers,
                            rows=rows,
                            title=f"Table {t_idx+1} on Page {page_num}"
                        ))
            except Exception:
                pass

            # Fallback label if page is visual only
            if not text and page_images:
                text = f"(Page {page_num} contains visual diagrams or figures without selectable text)"
            elif not text:
                text = f"(Page {page_num} is blank or contains no selectable text)"

            units.append(DocumentUnit(
                unit_number=page_num,
                unit_type="Page",
                text=text,
                heading=page_heading or f"Page {page_num}",
                images=page_images,
                tables=page_tables
            ))

        doc.close()

        return DocumentExtractionResult(
            filename=filename,
            file_type="PDF Document",
            file_size_bytes=file_size,
            units=units,
            headings=headings
        )

    @staticmethod
    def _extract_pptx(filename: str, file_bytes: bytes, file_size: int) -> DocumentExtractionResult:
        stream = io.BytesIO(file_bytes)
        prs = Presentation(stream)
        units: List[DocumentUnit] = []
        headings: List[str] = []
        seen_headings = set()

        # Track slide occurrences to filter out repeating slide master logos
        image_slide_counts: Dict[str, Set[int]] = {}
        for idx, slide in enumerate(prs.slides):
            for shape in slide.shapes:
                if hasattr(shape, "image"):
                    try:
                        h = hashlib.sha256(shape.image.blob).hexdigest()
                        if h not in image_slide_counts:
                            image_slide_counts[h] = set()
                        image_slide_counts[h].add(idx + 1)
                    except Exception:
                        pass

        repeating_logos = {
            h for h, slides in image_slide_counts.items()
            if len(slides) > 2 and len(prs.slides) >= 3
        }

        img_counter = 1

        for i, slide in enumerate(prs.slides):
            slide_num = i + 1
            slide_texts = []
            slide_title = None
            slide_images: List[ExtractedImage] = []
            slide_tables: List[ExtractedTable] = []

            for shape in slide.shapes:
                # Text content
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            if not slide_title and len(text) < 80:
                                slide_title = text
                            slide_texts.append(text)

                # Embedded images strictly on this slide
                if hasattr(shape, "image"):
                    try:
                        img_blob = shape.image.blob
                        img_hash = hashlib.sha256(img_blob).hexdigest()
                        if img_hash in repeating_logos:
                            continue

                        img_ext = shape.image.ext or "png"
                        with Image.open(io.BytesIO(img_blob)) as pil_img:
                            w, h = pil_img.size

                        if DocumentExtractor._is_valid_content_image(w, h):
                            slide_images.append(ExtractedImage(
                                image_id=f"pptx_img_{img_counter}",
                                unit_number=slide_num,
                                unit_type="Slide",
                                image_bytes=img_blob,
                                image_format=img_ext,
                                width=w,
                                height=h,
                                caption=f"Slide {slide_num} Figure ({w}x{h})"
                            ))
                            img_counter += 1
                    except Exception:
                        pass

                # Structured tables
                if shape.has_table:
                    try:
                        tbl = shape.table
                        t_rows = []
                        for row in tbl.rows:
                            row_cells = [cell.text.strip() for cell in row.cells]
                            t_rows.append(row_cells)
                        if t_rows:
                            headers = t_rows[0]
                            data_rows = t_rows[1:] if len(t_rows) > 1 else []
                            slide_tables.append(ExtractedTable(
                                table_id=f"pptx_tbl_{slide_num}",
                                unit_number=slide_num,
                                unit_type="Slide",
                                headers=headers,
                                rows=data_rows,
                                title=f"Slide {slide_num} Table"
                            ))
                    except Exception:
                        pass

            # Speaker Notes
            speaker_notes = None
            if slide.has_notes_slide:
                notes_frame = slide.notes_slide.notes_text_frame
                if notes_frame:
                    speaker_notes = notes_frame.text.strip()

            full_text = "\n".join(slide_texts) if slide_texts else "(Visual / Blank Slide)"
            if slide_title and slide_title not in seen_headings:
                headings.append(slide_title)
                seen_headings.add(slide_title)

            units.append(DocumentUnit(
                unit_number=slide_num,
                unit_type="Slide",
                text=full_text,
                heading=slide_title or f"Slide {slide_num}",
                images=slide_images,
                tables=slide_tables,
                speaker_notes=speaker_notes
            ))

        if not units:
            units.append(DocumentUnit(
                unit_number=1,
                unit_type="Slide",
                text="(Empty Presentation)",
                heading="Presentation"
            ))

        return DocumentExtractionResult(
            filename=filename,
            file_type="PowerPoint Presentation",
            file_size_bytes=file_size,
            units=units,
            headings=headings
        )

    @staticmethod
    def _extract_docx(filename: str, file_bytes: bytes, file_size: int) -> DocumentExtractionResult:
        stream = io.BytesIO(file_bytes)
        doc = DocxDocument(stream)
        units: List[DocumentUnit] = []
        headings: List[str] = []
        seen_headings = set()

        # Extract structured tables from docx
        doc_tables: List[ExtractedTable] = []
        for t_idx, table in enumerate(doc.tables):
            table_rows = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                table_rows.append(row_cells)
            if table_rows:
                headers = table_rows[0]
                data_rows = table_rows[1:] if len(table_rows) > 1 else []
                doc_tables.append(ExtractedTable(
                    table_id=f"docx_tbl_{t_idx + 1}",
                    unit_number=t_idx + 1,
                    unit_type="Section",
                    headers=headers,
                    rows=data_rows,
                    title=f"Document Table {t_idx + 1}"
                ))

        # Track which paragraph contains which image strictly via XML r:embed / r:id
        img_counter = 1
        current_heading = "Overview"
        current_paras: List[str] = []
        current_images: List[ExtractedImage] = []
        section_idx = 1

        for p in doc.paragraphs:
            text = p.strip() if hasattr(p, "strip") else p.text.strip()

            # Check if this paragraph contains an actually referenced body image
            p_xml = p._element.xml
            drawings = re.findall(r'<w:drawing[^>]*>.*?</w:drawing>', p_xml, re.DOTALL) or [p_xml] if ('embed="' in p_xml or 'id="' in p_xml) else []
            for d_xml in drawings:
                embed_match = re.search(r'embed="([^"]+)"', d_xml) or re.search(r'id="([^"]+)"', d_xml)
                if not embed_match:
                    continue
                eid = embed_match.group(1)
                part = doc.part.related_parts.get(eid)
                if part and hasattr(part, "content_type") and "image" in part.content_type:
                    try:
                        img_bytes = part.blob
                        with Image.open(io.BytesIO(img_bytes)) as pil_img:
                            w, h = pil_img.size
                            fmt = (pil_img.format or "png").lower()

                            # If Word applied an internal crop rectangle (<a:srcRect>), apply it strictly!
                            sr_match = re.search(r'<a:srcRect([^>]*)/>', d_xml)
                            if sr_match:
                                sr_attrs = sr_match.group(1)
                                l_m = re.search(r'l="(\d+)"', sr_attrs)
                                t_m = re.search(r't="(\d+)"', sr_attrs)
                                r_m = re.search(r'r="(\d+)"', sr_attrs)
                                b_m = re.search(r'b="(\d+)"', sr_attrs)
                                l = int(l_m.group(1)) if l_m else 0
                                t = int(t_m.group(1)) if t_m else 0
                                r = int(r_m.group(1)) if r_m else 0
                                b = int(b_m.group(1)) if b_m else 0

                                if l or t or r or b:
                                    left_px = max(0, int(w * l / 100000))
                                    top_px = max(0, int(h * t / 100000))
                                    right_px = min(w, int(w * (100000 - r) / 100000))
                                    bottom_px = min(h, int(h * (100000 - b) / 100000))
                                    if right_px > left_px and bottom_px > top_px:
                                        pil_img = pil_img.crop((left_px, top_px, right_px, bottom_px))
                                        w, h = pil_img.size

                            # Re-encode cropped image to bytes
                            buf = io.BytesIO()
                            save_fmt = "PNG" if fmt in ["png", "webp", "gif"] else "JPEG"
                            pil_img.save(buf, format=save_fmt)
                            cropped_bytes = buf.getvalue()

                        # Accept valid figures and formula diagrams (>= 40x30)
                        if w >= 40 and h >= 30:
                            current_images.append(ExtractedImage(
                                image_id=f"docx_img_{img_counter}",
                                unit_number=section_idx,
                                unit_type="Section",
                                image_bytes=cropped_bytes,
                                image_format=save_fmt.lower(),
                                width=w,
                                height=h,
                                caption=f"Figure {img_counter} ({w}x{h})"
                            ))
                            img_counter += 1
                    except Exception:
                        pass

            style_name = p.style.name if p.style else ""
            is_heading = (style_name and style_name.lower().startswith("heading")) or (
                len(text) < 70 and (
                    bool(re.match(r'^(chapter|unit|section|part)\s+\d+[:.\s\-]*$', text, re.IGNORECASE)) or
                    bool(re.match(r'^(chapter|unit|section|part)\s+\d+[:.\-]\s*[A-Z]', text, re.IGNORECASE)) or
                    bool(re.match(r'^\d+(\.\d+)*\s+[A-Z]', text)) or
                    text.startswith("#")
                )
            )

            if is_heading:
                if current_paras or current_images:
                    units.append(DocumentUnit(
                        unit_number=section_idx,
                        unit_type="Section",
                        text="\n\n".join(current_paras) if current_paras else "(Visual Section)",
                        heading=current_heading,
                        images=current_images
                    ))
                    section_idx += 1
                    current_paras = []
                    current_images = []

                current_heading = text
                if text not in seen_headings:
                    headings.append(text)
                    seen_headings.add(text)
            else:
                current_paras.append(text)

        if current_paras or current_images:
            units.append(DocumentUnit(
                unit_number=section_idx,
                unit_type="Section",
                text="\n\n".join(current_paras) if current_paras else "(Visual Section)",
                heading=current_heading,
                images=current_images
            ))

        if not units:
            units.append(DocumentUnit(
                unit_number=1,
                unit_type="Section",
                text="(No text extracted from document)",
                heading="Overview"
            ))

        # Attach doc tables to the first unit if available
        if units and doc_tables:
            units[0].tables = doc_tables

        return DocumentExtractionResult(
            filename=filename,
            file_type="Word DOCX Document",
            file_size_bytes=file_size,
            units=units,
            headings=headings
        )

    @staticmethod
    def _extract_standalone_image(filename: str, file_bytes: bytes, file_size: int) -> DocumentExtractionResult:
        with Image.open(io.BytesIO(file_bytes)) as pil_img:
            w, h = pil_img.size
            fmt = pil_img.format or "PNG"
            color_mode = pil_img.mode

        img_obj = ExtractedImage(
            image_id="uploaded_img_1",
            unit_number=1,
            unit_type="Image",
            image_bytes=file_bytes,
            image_format=fmt.lower(),
            width=w,
            height=h,
            caption=f"{filename} ({w}x{h}, {color_mode})"
        )

        unit = DocumentUnit(
            unit_number=1,
            unit_type="Image",
            text=f"Uploaded Image: {filename}\nResolution: {w}x{h} px\nFormat: {fmt}\nColor Mode: {color_mode}",
            heading=filename,
            images=[img_obj]
        )

        return DocumentExtractionResult(
            filename=filename,
            file_type=f"Image ({fmt})",
            file_size_bytes=file_size,
            units=[unit],
            headings=[filename]
        )

    @staticmethod
    def _extract_text(filename: str, file_bytes: bytes, file_size: int) -> DocumentExtractionResult:
        content = file_bytes.decode("utf-8", errors="replace")
        paragraphs = content.split("\n\n")
        units: List[DocumentUnit] = []
        headings: List[str] = []
        seen_headings = set()

        current_heading = "Introduction"
        current_paras = []
        section_idx = 1

        for p in paragraphs:
            text = p.strip()
            if not text:
                continue

            is_heading = text.startswith("#") or (re.match(r'^(chapter|unit|section|\d+\.)\b', text, re.IGNORECASE) and len(text) < 90)
            if is_heading:
                clean_h = text.lstrip("#").strip()
                if current_paras:
                    units.append(DocumentUnit(
                        unit_number=section_idx,
                        unit_type="Section",
                        text="\n\n".join(current_paras),
                        heading=current_heading
                    ))
                    section_idx += 1
                    current_paras = []
                current_heading = clean_h
                if clean_h not in seen_headings:
                    headings.append(clean_h)
                    seen_headings.add(clean_h)
            else:
                current_paras.append(text)

        if current_paras:
            units.append(DocumentUnit(
                unit_number=section_idx,
                unit_type="Section",
                text="\n\n".join(current_paras),
                heading=current_heading
            ))

        if not units:
            units.append(DocumentUnit(
                unit_number=1,
                unit_type="Section",
                text=content or "(Empty file)",
                heading="Content"
            ))

        return DocumentExtractionResult(
            filename=filename,
            file_type="Plain Text / Markdown",
            file_size_bytes=file_size,
            units=units,
            headings=headings
        )
