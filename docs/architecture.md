# Architecture

The rendered diagram is [`architecture.svg`](architecture.svg). The Mermaid source below is the
editable version of the same design.

## 1. Pipeline diagram

```mermaid
flowchart TB
    subgraph INDEX["INDEXING PIPELINE — once per uploaded document"]
        direction LR
        A["1 · PDF Upload<br/><i>app.py</i><br/>multi-file, ≤ 20 MB"]
        B["2 · Text Extraction<br/><i>pdf_loader.py</i><br/>per page + cleaning"]
        OCR["2ᵇ · OCR fallback<br/><i>ocr.py</i><br/>pdf2image → Tesseract<br/>only if page text &lt; 80 chars"]
        C["3 · Text Chunking<br/><i>chunker.py</i><br/>recursive, page-scoped"]
        D["4 · Embedding Generation<br/><i>embeddings.py</i><br/>all-MiniLM-L6-v2, 384-d"]
        E["5 · Vector Upsert<br/><i>vector_store.py</i><br/>batched, namespaced"]
        A --> B
        B -. "page has no text" .-> OCR
        OCR -. "merged back per page" .-> B
        B --> C --> D --> E
    end

    PC[("PINECONE SERVERLESS INDEX<br/>metric: cosine · dim: 384 · aws/us-east-1<br/>namespace per corpus<br/>metadata: document_name, page_number,<br/>chunk_id, chunk_index, text")]

    subgraph QUERY["QUERY PIPELINE — every question"]
        direction LR
        F["6 · User Question<br/><i>app.py</i><br/>+ top-k, threshold, filters"]
        G["7 · Query Embedding<br/><i>embeddings.py</i><br/>same model as indexing"]
        H["8 · Semantic Retrieval<br/><i>retriever.py</i><br/>top-k · threshold · dedupe"]
        I["9 · LLM Generation<br/><i>generator.py</i><br/>context-only prompt, T=0"]
        J["10 · Answer + Sources<br/><i>app.py</i><br/>page · excerpt · similarity"]
        F --> G --> H --> I --> J
    end

    E -->|upsert| PC
    H -->|query vector| PC
    PC -->|top-k matches + metadata| H
```

## 2. Module dependency graph

Dependencies run in one direction only; nothing in `src/` imports Streamlit.

```mermaid
flowchart LR
    APP["app.py<br/>(Streamlit)"] --> PIPE["pipeline.py"]
    APP --> OCR["ocr.py"]
    PIPE --> LOAD["pdf_loader.py"]
    LOAD --> OCR
    OCR --> CFG
    PIPE --> CHUNK["chunker.py"]
    PIPE --> EMB["embeddings.py"]
    PIPE --> VS["vector_store.py"]
    PIPE --> RET["retriever.py"]
    PIPE --> GEN["generator.py"]
    PIPE --> LOG["query_logger.py"]
    RET --> VS
    RET --> EMB
    LOAD --> M["models.py"]
    CHUNK --> M
    RET --> M
    GEN --> M
    LOAD --> CFG["config.py"]
    EMB --> CFG
    VS --> CFG
    GEN --> CFG
    PIPE --> CFG
```

## 3. Data flow

### 3.1 Indexing

| Step | Input | Output | Owner |
|---|---|---|---|
| Validate | uploaded bytes + filename | — (raises `PDFProcessingError`) | `pdf_loader.validate_pdf_bytes` |
| Extract | PDF bytes | `Document(pages=[PageText(page_number, text, source)])` | `pdf_loader.load_pdf` |
| Clean | raw page text | de-hyphenated, unwrapped, furniture-free text | `pdf_loader.clean_text` |
| Detect scans | cleaned page text | page numbers below `OCR_MIN_CHARS` | `ocr.needs_ocr` |
| OCR *(only those pages)* | PDF bytes + page numbers | `{page_number: raw_text}` | `ocr.ocr_page_numbers` |
| Merge | native text + OCR text | one page text, de-duplicated | `ocr.merge_page_text` |
| Chunk | `Document` | `list[Chunk]` with `chunk_id`, `page_number`, `chunk_index` | `chunker.chunk_document` |
| Embed | chunk texts | `ndarray (n, 384)`, L2-normalised | `embeddings.embed_documents` |
| Upsert | chunks + vectors + namespace | vectors in Pinecone | `vector_store.upsert_chunks` |

### 3.2 Querying

| Step | Input | Output | Owner |
|---|---|---|---|
| Embed query | question text | `ndarray (384,)` | `embeddings.embed_query` |
| Build filter | selected docs, page range | Pinecone filter dict (`$in`, `$gte`/`$lte`) | `vector_store.build_metadata_filter` |
| Search | query vector, namespace, filter | raw matches (over-fetched 3×) | `vector_store.query` |
| Gate | matches | `RetrievalResult` (threshold applied, de-duplicated, trimmed to top-k) | `retriever.retrieve` |
| Overview fallback *(only if the gate emptied and the query is a meta-question)* | query + filters | opening chunks (`chunk_index < 4`) in reading order | `retriever._retrieve_overview` |
| Generate | `RetrievalResult` | `Answer` (text, sources, confidence, grounded) | `generator.generate` |
| Log | result + answer | one JSONL record | `query_logger.log_query` |
| Render | `Answer` | answer, confidence, per-source page/excerpt/score | `app.render_answer` |

## 4. Key design points

**Page-scoped chunking.** Chunks never span a page boundary. This costs a little context at page
breaks but guarantees that every chunk has one unambiguous page number — which is what makes the
source attribution trustworthy rather than approximate.

**Namespaces over indexes.** The Pinecone free tier allows a limited number of indexes but
unlimited namespaces. Using one namespace per corpus gives isolation between document sets and
sessions at no quota cost, and makes "clear my corpus" a single `delete(delete_all=True)` call.

**Normalised vectors.** All embeddings are L2-normalised before upsert, so cosine similarity and
dot product agree and scores fall in an interpretable range — necessary for the similarity
threshold to mean the same thing across models.

**Over-fetch then filter.** Retrieval asks Pinecone for `3 × top_k` candidates, then applies the
threshold and de-duplication locally before trimming to `top_k`. Without this, near-identical
overlapping chunks would consume the context budget and crowd out genuinely distinct evidence.

**OCR is a fallback, not a stage.** OCR sits *inside* extraction rather than as a pipeline step,
because it only ever runs for the subset of pages that produced no text. A text PDF's path through
the system is byte-for-byte what it was before OCR existed, and the toolchain being absent
degrades to a warning rather than an error — OCR is an enhancement, never a dependency.

**Two retrieval modes, not one threshold.** Similarity search answers questions about *content*.
A question about the *document itself* ("what is this about?") scores near zero against that
document — 0.082, against 0.040 for a wholly unrelated question — so no threshold can separate
them. Those queries are served instead by a metadata-filtered query for the opening chunks. It
fires only when similarity found nothing *and* the phrasing is recognised as a meta-question, so
genuinely unanswerable questions are still refused.

**Three-point hallucination defence.** A retrieval gate (no context → no LLM call), a strict
context-only prompt at `temperature=0`, and a post-generation refusal-and-citation check. Each
works even if the other two fail.
