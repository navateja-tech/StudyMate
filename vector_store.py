import os
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Plus
from extractor import DocumentExtractionResult, DocumentUnit, ExtractedImage

class TextChunk:
    """Represents a context-preserved chunk of text from a document."""
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        filename: str,
        unit_number: int,
        unit_type: str,
        heading: str,
        raw_text: str,
        context_text: str,
        chunk_index: int,
        total_chunks: int = 1,
        prev_chunk_id: Optional[str] = None,
        next_chunk_id: Optional[str] = None,
        has_images: bool = False,
        image_ids: Optional[List[str]] = None,
        has_tables: bool = False,
        table_ids: Optional[List[str]] = None
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.filename = filename
        self.unit_number = unit_number
        self.unit_type = unit_type
        self.heading = heading
        self.raw_text = raw_text.strip()
        self.context_text = context_text.strip()
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.prev_chunk_id = prev_chunk_id
        self.next_chunk_id = next_chunk_id
        self.has_images = has_images
        self.image_ids = image_ids or []
        self.has_tables = has_tables
        self.table_ids = table_ids or []
        self.word_count = len(self.raw_text.split())

    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk metadata to Chroma-compatible flat dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "unit_number": int(self.unit_number),
            "unit_type": self.unit_type,
            "heading": self.heading or "General",
            "chunk_index": int(self.chunk_index),
            "total_chunks": int(self.total_chunks),
            "word_count": int(self.word_count),
            "has_images": bool(self.has_images),
            "image_count": len(self.image_ids),
            "has_tables": bool(self.has_tables),
            "table_count": len(self.table_ids),
            "prev_chunk_id": self.prev_chunk_id or "",
            "next_chunk_id": self.next_chunk_id or ""
        }


class ContextPreservingChunker:
    """
    Intelligent document chunker that maintains hierarchical and contextual
    integrity to prevent semantic loss during vector retrieval.
    """

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences while respecting abbreviations and punctuation."""
        sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z0-9#\-\*])', text)
        cleaned = [s.strip() for s in sentences if s.strip()]
        return cleaned if cleaned else [text]

    @classmethod
    def chunk_document(
        cls,
        doc_result: DocumentExtractionResult,
        target_chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[TextChunk]:
        """
        Chunks an extracted document while injecting hierarchical context
        into each chunk (document title, unit type, unit number, heading).
        """
        all_chunks: List[TextChunk] = []
        doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_result.filename)
        global_chunk_idx = 0

        for unit in doc_result.units:
            unit_text = unit.text.strip()
            if not unit_text:
                continue

            unit_heading = unit.heading or f"{unit.unit_type} {unit.unit_number}"
            img_ids = [img.image_id for img in getattr(unit, "images", [])]
            tbl_ids = [tbl.table_id for tbl in getattr(unit, "tables", [])]
            has_img = len(img_ids) > 0
            has_tbl = len(tbl_ids) > 0

            # Short units kept as a single chunk to maintain topical cohesion
            if len(unit_text) <= (target_chunk_size + chunk_overlap):
                chunk_id = f"{doc_id}_c{global_chunk_idx + 1}"
                
                # Context Breadcrumb Header: Preserves topic and structural context
                context_header = f"[Source: {doc_result.filename} | {unit.unit_type} {unit.unit_number} | Topic: {unit_heading}]"
                context_full = f"{context_header}\n{unit_text}"

                chunk = TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=doc_result.filename,
                    unit_number=unit.unit_number,
                    unit_type=unit.unit_type,
                    heading=unit_heading,
                    raw_text=unit_text,
                    context_text=context_full,
                    chunk_index=global_chunk_idx + 1,
                    has_images=has_img,
                    image_ids=img_ids,
                    has_tables=has_tbl,
                    table_ids=tbl_ids
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1
            else:
                # Segment long unit into overlapping sentence-aware windows
                sentences = cls._split_into_sentences(unit_text)
                current_window = []
                current_length = 0
                unit_sub_chunks = []

                for sentence in sentences:
                    sentence_len = len(sentence)
                    if current_length + sentence_len > target_chunk_size and current_window:
                        chunk_raw = " ".join(current_window).strip()
                        unit_sub_chunks.append(chunk_raw)

                        overlap_window = []
                        overlap_len = 0
                        for s in reversed(current_window):
                            if overlap_len + len(s) <= chunk_overlap:
                                overlap_window.insert(0, s)
                                overlap_len += len(s)
                            else:
                                break
                        current_window = overlap_window + [sentence]
                        current_length = sum(len(s) for s in current_window)
                    else:
                        current_window.append(sentence)
                        current_length += sentence_len

                if current_window:
                    chunk_raw = " ".join(current_window).strip()
                    if chunk_raw and (not unit_sub_chunks or chunk_raw != unit_sub_chunks[-1]):
                        unit_sub_chunks.append(chunk_raw)

                # Build TextChunk objects with breadcrumb context
                for sub_idx, sub_text in enumerate(unit_sub_chunks):
                    chunk_id = f"{doc_id}_c{global_chunk_idx + 1}"
                    sub_label = f"Part {sub_idx + 1}" if len(unit_sub_chunks) > 1 else ""
                    context_header = f"[Source: {doc_result.filename} | {unit.unit_type} {unit.unit_number} ({sub_label}) | Topic: {unit_heading}]"
                    context_full = f"{context_header}\n{sub_text}"

                    chunk = TextChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        filename=doc_result.filename,
                        unit_number=unit.unit_number,
                        unit_type=unit.unit_type,
                        heading=unit_heading,
                        raw_text=sub_text,
                        context_text=context_full,
                        chunk_index=global_chunk_idx + 1,
                        has_images=has_img,
                        image_ids=img_ids,
                        has_tables=has_tbl,
                        table_ids=tbl_ids
                    )
                    all_chunks.append(chunk)
                    global_chunk_idx += 1

        # Link adjacent chunks for contextual expansion during RAG
        total = len(all_chunks)
        for i, chk in enumerate(all_chunks):
            chk.total_chunks = total
            if i > 0:
                chk.prev_chunk_id = all_chunks[i - 1].chunk_id
            if i < total - 1:
                chk.next_chunk_id = all_chunks[i + 1].chunk_id

        return all_chunks


class StudyMateVectorStore:
    """
    Advanced Knowledge Store featuring:
    1. Dense Semantic Embeddings (SentenceTransformers all-MiniLM-L6-v2 in ChromaDB)
    2. BM25 Lexical Keyword Search (BM25Plus with technical tokenization)
    3. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
    4. Context Window Expansion (Surrounding Neighbor Stitching)
    """
    _instance = None

    def __init__(self, persist_dir: str = "./data/chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.persist_dir = os.path.abspath(persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.model_name = model_name

        # Initialize SentenceTransformer
        try:
            self.embedder = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.embedder = SentenceTransformer(model_name)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection_name = "studymate_knowledge"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # In-Memory BM25 Index for Hybrid Retrieval
        self.bm25_chunk_ids: List[str] = []
        self.bm25_corpus: List[List[str]] = []
        self.bm25_index: Optional[BM25Plus] = None

        # Build initial BM25 index from existing Chroma entries
        self._sync_bm25_index()

    @classmethod
    def get_instance(cls, persist_dir: str = "./data/chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = StudyMateVectorStore(persist_dir=persist_dir, model_name=model_name)
        return cls._instance

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text preserving technical hyphenated terms and acronyms."""
        return re.findall(r'[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*', (text or "").lower())

    def _sync_bm25_index(self):
        """Synchronize the BM25 index with current items in the Chroma collection."""
        try:
            count = self.collection.count()
            if count == 0:
                self.bm25_chunk_ids = []
                self.bm25_corpus = []
                self.bm25_index = None
                return

            all_records = self.collection.get(include=["documents", "metadatas"])
            if all_records and all_records.get("ids"):
                self.bm25_chunk_ids = all_records["ids"]
                self.bm25_corpus = [
                    self._tokenize(doc + " " + meta.get("heading", "") + " " + meta.get("filename", ""))
                    for doc, meta in zip(all_records["documents"], all_records["metadatas"])
                ]
                self.bm25_index = BM25Plus(self.bm25_corpus)
        except Exception:
            self.bm25_index = None

    def index_document(
        self,
        doc_result: DocumentExtractionResult,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> Dict[str, Any]:
        """
        Chunks the document, generates dense embeddings, saves into ChromaDB,
        and synchronizes the BM25 lexical index.
        """
        chunks = ContextPreservingChunker.chunk_document(
            doc_result,
            target_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not chunks:
            return {"status": "empty", "chunks_indexed": 0, "filename": doc_result.filename}

        # Remove existing entries for this filename to prevent duplicate accumulation
        self.delete_document(doc_result.filename)

        ids = [c.chunk_id for c in chunks]
        documents_to_embed = [c.context_text for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        # Generate dense embeddings
        embeddings = self.embedder.encode(
            documents_to_embed,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()

        # Insert into ChromaDB collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[c.raw_text for c in chunks],
            metadatas=metadatas
        )

        # Synchronize BM25 lexical index
        self._sync_bm25_index()

        return {
            "status": "success",
            "filename": doc_result.filename,
            "chunks_indexed": len(chunks),
            "model_name": self.model_name,
            "dimensions": len(embeddings[0]) if embeddings else 384,
            "chunks": chunks
        }

    def expand_chunk_context(self, chunk_id: str, window_size: int = 1) -> Dict[str, Any]:
        """
        Context Window Expansion: Stitches preceding and subsequent adjacent chunks
        to provide complete surrounding narrative context without truncation.
        """
        record = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not record or not record.get("ids"):
            return {"core_text": "", "expanded_text": "", "surrounding_ids": []}

        core_text = record["documents"][0]
        meta = record["metadatas"][0]
        filename = meta.get("filename")

        prev_texts = []
        curr_prev_id = meta.get("prev_chunk_id")
        steps = 0
        while curr_prev_id and steps < window_size:
            prev_rec = self.collection.get(ids=[curr_prev_id], include=["documents", "metadatas"])
            if prev_rec and prev_rec.get("ids"):
                p_text = prev_rec["documents"][0]
                p_meta = prev_rec["metadatas"][0]
                if p_meta.get("filename") == filename:
                    prev_texts.insert(0, (curr_prev_id, p_meta, p_text))
                    curr_prev_id = p_meta.get("prev_chunk_id")
                    steps += 1
                else:
                    break
            else:
                break

        next_texts = []
        curr_next_id = meta.get("next_chunk_id")
        steps = 0
        while curr_next_id and steps < window_size:
            next_rec = self.collection.get(ids=[curr_next_id], include=["documents", "metadatas"])
            if next_rec and next_rec.get("ids"):
                n_text = next_rec["documents"][0]
                n_meta = next_rec["metadatas"][0]
                if n_meta.get("filename") == filename:
                    next_texts.append((curr_next_id, n_meta, n_text))
                    curr_next_id = n_meta.get("next_chunk_id")
                    steps += 1
                else:
                    break
            else:
                break

        # Assemble formatted contiguous stitched passage
        parts = []
        surrounding_ids = []

        if prev_texts:
            for p_id, p_m, p_t in prev_texts:
                parts.append(f"⏮️ [Preceding Context • {p_m.get('unit_type')} {p_m.get('unit_number')}: {p_m.get('heading')}]\n{p_t}")
                surrounding_ids.append(p_id)

        parts.append(f"🎯 [Matched Passage • {meta.get('unit_type')} {meta.get('unit_number')}: {meta.get('heading')}]\n{core_text}")

        if next_texts:
            for n_id, n_m, n_t in next_texts:
                parts.append(f"⏭️ [Following Context • {n_m.get('unit_type')} {n_m.get('unit_number')}: {n_m.get('heading')}]\n{n_t}")
                surrounding_ids.append(n_id)

        expanded_text = "\n\n".join(parts)

        return {
            "core_text": core_text,
            "expanded_text": expanded_text,
            "surrounding_ids": surrounding_ids,
            "has_expanded": len(prev_texts) > 0 or len(next_texts) > 0
        }

    def search_dense(
        self,
        query: str,
        top_k: int = 10,
        filename_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Dense semantic vector search using ChromaDB."""
        if not query.strip() or self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode(
            [query.strip()],
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()

        where_filter = {"filename": filename_filter} if filename_filter else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        matches = []
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][idx]
                raw_text = results["documents"][0][idx]
                metadata = results["metadatas"][0][idx]
                distance = results["distances"][0][idx]
                similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                matches.append({
                    "chunk_id": chunk_id,
                    "text": raw_text,
                    "metadata": metadata,
                    "distance": round(distance, 4),
                    "score": similarity,
                    "source": "dense"
                })

        return matches

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        filename_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Sparse lexical keyword search using BM25Plus."""
        if not query.strip() or not self.bm25_index or not self.bm25_chunk_ids:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25_index.get_scores(tokens)
        scored_pairs = list(zip(self.bm25_chunk_ids, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        matches = []
        max_score = scored_pairs[0][1] if scored_pairs and scored_pairs[0][1] > 0 else 1.0

        for chunk_id, score in scored_pairs[:top_k * 2]:
            if score <= 0:
                continue
            rec = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            if not rec or not rec.get("ids"):
                continue
            meta = rec["metadatas"][0]
            if filename_filter and meta.get("filename") != filename_filter:
                continue

            norm_score = min(1.0, score / max_score)
            matches.append({
                "chunk_id": chunk_id,
                "text": rec["documents"][0],
                "metadata": meta,
                "bm25_score": round(score, 3),
                "score": norm_score,
                "source": "bm25"
            })
            if len(matches) >= top_k:
                break

        return matches

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filename_filter: Optional[str] = None,
        alpha: float = 0.6,
        expand_context: bool = True,
        window_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Search combining Dense Vector Search + BM25 Keyword Search
        using Reciprocal Rank Fusion (RRF) and Context Window Expansion.
        """
        if not query.strip() or self.collection.count() == 0:
            return []

        candidate_k = max(top_k * 3, 10)
        dense_results = self.search_dense(query, top_k=candidate_k, filename_filter=filename_filter)
        bm25_results = self.search_bm25(query, top_k=candidate_k, filename_filter=filename_filter)

        # Reciprocal Rank Fusion (RRF constant = 60)
        RRF_K = 60
        rrf_scores: Dict[str, float] = {}
        items_map: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}

        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            dense_ranks[cid] = rank + 1
            items_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (alpha / (RRF_K + rank + 1))

        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            bm25_ranks[cid] = rank + 1
            if cid not in items_map:
                items_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + ((1.0 - alpha) / (RRF_K + rank + 1))

        # Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
        max_rrf = rrf_scores[sorted_cids[0]] if sorted_cids else 1.0

        results = []
        for cid in sorted_cids[:top_k]:
            item = items_map[cid]
            raw_rrf = rrf_scores[cid]
            # Normalize RRF to 0-100 percentage
            hybrid_confidence = round((raw_rrf / max_rrf) * 100, 1)

            entry = {
                "chunk_id": cid,
                "text": item["text"],
                "metadata": item["metadata"],
                "hybrid_score": hybrid_confidence,
                "dense_rank": dense_ranks.get(cid, "-"),
                "bm25_rank": bm25_ranks.get(cid, "-"),
                "raw_rrf": round(raw_rrf, 5)
            }

            # Apply Context Window Expansion if requested
            if expand_context:
                expanded_data = self.expand_chunk_context(cid, window_size=window_size)
                entry["expanded_text"] = expanded_data["expanded_text"]
                entry["has_expanded"] = expanded_data["has_expanded"]
                entry["surrounding_ids"] = expanded_data["surrounding_ids"]
            else:
                entry["expanded_text"] = item["text"]
                entry["has_expanded"] = False
                entry["surrounding_ids"] = []

            results.append(entry)

        return results

    # Legacy alias
    def search(self, query: str, top_k: int = 5, filename_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.hybrid_search(query, top_k=top_k, filename_filter=filename_filter)

    def get_document_chunks(self, filename: str) -> List[Dict[str, Any]]:
        """Retrieve all stored chunks for a specific document."""
        if self.collection.count() == 0:
            return []

        results = self.collection.get(
            where={"filename": filename},
            include=["documents", "metadatas"]
        )

        chunks = []
        if results and results.get("ids"):
            for idx in range(len(results["ids"])):
                chunks.append({
                    "chunk_id": results["ids"][idx],
                    "text": results["documents"][idx],
                    "metadata": results["metadatas"][idx]
                })

        chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        return chunks

    def delete_document(self, filename: str) -> int:
        """Deletes all chunks belonging to a filename and updates BM25 index."""
        try:
            existing = self.collection.get(where={"filename": filename})
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
                self._sync_bm25_index()
                return len(existing["ids"])
        except Exception:
            pass
        return 0

    def clear_all(self) -> int:
        """Completely purges all vectors, stored documents, and the BM25 index."""
        count = 0
        try:
            count = self.collection.count()
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_chunk_ids = []
        self.bm25_corpus = []
        self.bm25_index = None
        return count

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """Lists distinct indexed documents with chunk count."""
        if self.collection.count() == 0:
            return []

        all_meta = self.collection.get(include=["metadatas"])
        docs = {}
        for m in all_meta.get("metadatas", []):
            fn = m.get("filename", "Unknown")
            if fn not in docs:
                docs[fn] = {
                    "filename": fn,
                    "doc_id": m.get("doc_id", ""),
                    "total_chunks": 0,
                    "unit_type": m.get("unit_type", "Unit")
                }
            docs[fn]["total_chunks"] += 1

        return list(docs.values())

    def get_stats(self) -> Dict[str, Any]:
        """Returns collection stats."""
        total = self.collection.count()
        docs = self.list_indexed_documents()
        return {
            "total_chunks": total,
            "total_documents": len(docs),
            "model_name": self.model_name,
            "dimension": 384,
            "has_bm25": self.bm25_index is not None,
            "bm25_terms_indexed": len(self.bm25_corpus),
            "persist_directory": self.persist_dir
        }
