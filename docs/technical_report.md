# Technical Report — Intermediate RAG System Using Pinecone Vector Database

**Course:** Artificial Intelligence / NLP / Applied LLM Systems
**Level:** Intermediate
**Deliverable:** Retrieval-Augmented Generation system over user-supplied PDF documents

---

## 1. Introduction and objective

Large language models answer fluently whether or not they know the answer. The purpose of this
system is to remove that failure mode for a bounded corpus: a user uploads one or more PDFs, asks
questions in natural language, and receives answers that are derived *only* from those PDFs, with
every claim traceable to a page number and a similarity score. When the documents do not contain
the answer, the system says so in a fixed sentence rather than guessing.

The implementation follows the eight-stage pipeline required by the assignment — upload,
extraction, chunking, embedding, Pinecone indexing, semantic retrieval, LLM generation, and
attributed answer — with each stage isolated in its own module.

---

## 2. System design decisions

### 2.1 Layered, one-directional architecture

The interface (`app.py`) contains no domain logic, and no module under `src/` imports Streamlit.
Everything the interface needs is reached through a single façade, `RAGPipeline`. The dependency
direction is strictly `app.py → pipeline.py → {loader, chunker, embeddings, vector_store,
retriever, generator}`.

The practical payoff is testability and portability. The 33 unit tests exercise chunking,
cleaning, retrieval policy and confidence scoring without a browser, an API key or a network
connection, and swapping Streamlit for a FastAPI route would require touching exactly one file.

### 2.2 Page-scoped chunking

Chunks are built page by page and never span a page boundary. This is the single most
consequential design decision in the system, and it is a deliberate trade-off:

* **Cost:** a paragraph that straddles a page break is split, so a small amount of context is lost
  at each boundary.
* **Benefit:** every chunk carries exactly one page number. Source attribution is therefore exact
  rather than approximate — a chunk can never be cited as "pages 4–5".

Because the assignment grades source attribution and hallucination prevention, exactness was
judged more valuable than the marginal context recovered by cross-page chunks.

### 2.3 Recursive splitting with sentence-aligned overlap

Within a page, the splitter tries progressively finer natural boundaries — paragraph (`\n\n`),
line, sentence (`. `, `? `, `! `), clause (`; `, `, `), word — and only falls back to a hard
character cut when a single token exceeds the window. Separators are re-attached to the text they
followed, so splitting on `". "` does not silently delete sentence-ending punctuation.

Overlap is applied by prepending a tail of the previous chunk, trimmed forward to a sentence
boundary where one exists and to a word boundary otherwise. Two bugs found during development are
worth recording, because both produced *silently* degraded chunks rather than errors:

1. A tail that ended exactly on a sentence boundary produced an empty trailing element from
   `re.split`, which collapsed the overlap to nothing. Overlap was silently disabled for a large
   fraction of chunks.
2. Before separators were re-attached, chunk text read `"…model memory The retrieval step…"` — the
   period had been consumed by `str.split`, corrupting sentence structure for the embedder.

Both are now covered by regression tests (`test_overlap_applies_when_tail_ends_on_a_sentence_boundary`,
`test_sentence_punctuation_is_preserved_across_splits`).

### 2.4 Text cleaning

PDF text extraction produces predictable artifacts, each handled explicitly in
`pdf_loader.clean_text`:

| Artifact | Treatment |
|---|---|
| `retrie-\nval` hyphenation from justified typesetting | rejoined into one word |
| Single newlines inside sentences (line wraps) | converted to spaces; blank-line paragraph breaks preserved |
| Standalone page furniture (`12`, `- 12 -`, `Page 3 of 40`) | dropped line-wise |
| Ligature glyphs (`ﬁ`, `ﬂ`, `ﬀ`) | expanded to ASCII |
| Control characters and `` replacement glyphs | stripped |
| Runs of spaces, tabs, non-breaking and zero-width spaces | collapsed |

This matters more than it appears: un-rejoined hyphenation turns one meaningful token into two
meaningless ones, and both then pollute the embedding.

### 2.5 OCR fallback for scanned documents

A scanned PDF contains images of text, not text. `pypdf` extracts nothing from it, so without
OCR the system would either reject the upload or — worse — index an empty document and then
refuse every question about it.

The fallback is deliberately **automatic and invisible**: `pdf_loader` extracts natively first,
then asks `ocr.needs_ocr()` which pages fell below `OCR_MIN_CHARS` (default 80) and hands only
those to Tesseract. A text PDF therefore never pays the OCR cost, and the user never chooses a
mode.

Three decisions are worth recording:

**Why a character threshold rather than an emptiness test.** Scanned pages frequently extract a
handful of characters from a watermark, a header stamp, or an OCR layer added by a previous tool.
`len(text) == 0` would miss those pages; 80 characters is comfortably below any real page of
prose and comfortably above that noise.

**Why merging is not concatenation.** OCR reads the *whole* rendered page, so on a mixed page it
re-reads the native text as well. Blind concatenation would duplicate that text into the chunker,
and the duplicate would then compete for slots in the retrieved context. `ocr.merge_page_text`
measures word coverage: if the OCR output already contains ≥ 80 % of the native words it replaces
the native text; otherwise the two are genuinely different content (a text paragraph beside a
scanned figure) and are concatenated.

**Why page-at-a-time rasterisation.** A 300 DPI A4 page is roughly a 25 MB bitmap. Converting a
whole scanned document at once would exhaust memory on a modest machine, so `ocr_page_numbers`
renders, reads, and closes one page at a time.

The toolchain — Tesseract and Poppler — consists of system binaries that pip cannot install, so
every entry point degrades rather than raises: `check_availability()` never throws, the sidebar
reports readiness, and a scanned upload without OCR produces installation guidance instead of a
stack trace. `OCR_ENABLED=false` disables the stage outright.

### 2.6 Error handling

Each stage raises its own typed exception — `PDFProcessingError`, `ChunkingError`,
`EmbeddingError`, `VectorStoreError`, `RetrievalError`, `GenerationError` — and `app.py` catches
them individually so the user sees an actionable message ("this PDF is a scan, run OCR first")
rather than a stack trace. Failures during multi-file upload are per-file: one corrupt PDF does
not abort the batch. Query logging failures are swallowed with a warning, because telemetry must
never break the user-facing path.

---

## 3. Embedding model

**Model:** `sentence-transformers/all-MiniLM-L6-v2` — 6-layer MiniLM, 384 dimensions, ~80 MB,
maximum sequence length 256 word-pieces.

**Why this model:**

* **Dimensionality.** At 384 dimensions it stores 2.7× more vectors per unit of Pinecone free-tier
  storage than a 1024-d model, with minor retrieval-quality loss on short-passage semantic search.
* **Cost.** It runs locally on CPU with no per-token charge, so indexing a large document is free
  and repeatable — important when tuning chunk size, which requires re-indexing.
* **Symmetry.** It encodes queries and passages with the same function, so no asymmetric
  `query:`/`passage:` prefixing is needed and a short question compares sensibly against a long
  chunk.
* **Fit to chunk size.** The 256 word-piece limit corresponds to roughly 1,000–1,200 characters of
  English prose, which brackets the default 800-character chunk. Chunks are therefore embedded
  whole rather than silently truncated. **Raising the chunk-size slider above ~1,200 characters
  will cause truncation** and is the main caveat on that control.

All vectors are **L2-normalised** before upsert. This makes cosine similarity and dot product
equivalent and keeps scores in an interpretable range, which is what allows a fixed similarity
threshold to behave consistently.

**Pluggable backend.** `embeddings.py` defines a `BaseEmbedder` interface with two
implementations. Setting `EMBEDDING_BACKEND=pinecone` switches to Pinecone's hosted
`multilingual-e5-large` (1024-d) without touching any other module — an escape hatch for
environments where local `torch` wheels are unavailable for the installed Python version (see
§6.5).

---

## 4. Pinecone configuration

### 4.1 Index

| Setting | Value | Rationale |
|---|---|---|
| Type | Serverless | No pod management, free tier, scales to zero |
| Cloud / region | `aws` / `us-east-1` | The free-tier serverless region |
| Metric | `cosine` | Required by the assignment; correct for normalised sentence embeddings |
| Dimension | 384 | Matches `all-MiniLM-L6-v2` |
| Creation | On demand at startup | `_ensure_index` creates the index if absent and polls `describe_index` until `status.ready`, with a 90 s timeout |

A dimension guard runs when the index already exists: if the stored dimension does not match the
configured embedding model, startup fails with an explicit message. Without it, a model change
produces an opaque upsert error hundreds of vectors later.

### 4.2 Namespaces

One namespace per corpus, defaulting to a timestamped session ID and editable in the sidebar.
Namespace strings are sanitised to `[A-Za-z0-9._-]`, lower-cased and truncated to 48 characters.

The free tier limits the number of *indexes* but not namespaces, so namespaces give isolation
between document sets at zero quota cost. They also make corpus deletion a single
`delete(delete_all=True, namespace=…)` call, and let a user return to a previous corpus simply by
typing its namespace name.

### 4.3 Upserts

Vectors are written in batches of 96 — comfortably inside Pinecone's 1,000-vector / 2 MB request
limits once ~800 characters of chunk text per vector are accounted for. Each batch retries up to
three times with linear backoff, which absorbs the transient rate-limit responses the free tier
returns under load. A progress callback reports batch completion to the UI.

### 4.4 Metadata

Every vector carries:

```json
{
  "chunk_id":      "paper.pdf::p7::c42::a1b2c3d4e5f6a7b8",
  "text":          "…the chunk text, returned for citation…",
  "document_name": "paper.pdf",
  "page_number":   7,
  "chunk_index":   42
}
```

Storing `text` in the metadata costs storage but removes an entire component from the system: no
external document store is needed to render a citation, because the excerpt comes back with the
match. Chunk IDs embed a SHA-1 fingerprint of the content plus the namespace, so re-uploading the
same document idempotently overwrites its vectors instead of duplicating them, and the readable
prefix is truncated to 64 characters to stay inside Pinecone's 512-byte ID limit.

### 4.5 Metadata filtering

`build_metadata_filter` composes Pinecone filter expressions from the UI selections:

```python
{"document_name": {"$in": ["a.pdf", "b.pdf"]},
 "page_number":   {"$gte": 4, "$lte": 10}}
```

Filtering happens server-side inside the ANN search, so a page-range filter narrows the candidate
set rather than discarding results after the fact.

---

## 5. Retrieval and generation strategy

### 5.1 Retrieval

Retrieval over-fetches `3 × top_k` candidates (capped at 60), then applies, in order: the
similarity threshold, shingle-based de-duplication, and a final trim to `top_k`.

De-duplication compares 8-word shingle sets and drops any chunk that is ≥ 80 % contained in a
higher-scoring one. This is necessary precisely *because* of chunk overlap: adjacent chunks share
text by design, so without de-duplication the top-k slots fill with the same sentences repeated
and genuinely distinct evidence is pushed out of the context window.

### 5.2 Hallucination prevention

Three independent guards, each of which works if the other two fail:

1. **Retrieval gate.** If no chunk clears the threshold, `generate()` returns the refusal sentence
   *without calling the LLM*. The model cannot invent an answer it was never asked for, and the
   failure costs no tokens.
2. **Prompt contract.** The system prompt forbids outside knowledge, requires an inline `[S#]`
   citation for every claim, mandates the exact refusal sentence, and instructs the model to state
   explicitly which part of a question the document does not cover. Context is supplied as
   numbered blocks tagged with document, page and similarity. Generation runs at
   `temperature=0.0`, `top_p=1.0`.
3. **Post-generation check.** The response is scanned for the refusal sentence and for `[S#]`
   markers. Only the sources the answer actually cited are displayed, and the citation rate feeds
   the confidence score, so an answer that ignores its context scores visibly lower.

### 5.3 The overview fallback — when similarity search structurally cannot work

Testing the finished system surfaced a failure that no amount of threshold tuning could fix.
Asked *"What is this document about?"* over a marine-biology report, the system refused. Measured
cosine scores against that document's own chunks:

| Query | Best cosine | Verdict at 0.35 |
|---|---|---|
| What threatens the Hawksbill turtle? | **0.683** | answered |
| By how much did nesting sites decline? | **0.400** | answered |
| What is this document about? | **0.082** | refused |
| What is the main topic? | 0.086 | refused |
| Summarise this document | 0.063 | refused |
| *What is the capital of France?* (unrelated) | *0.040* | refused |

The meta-question scores **0.082 — barely above an entirely unrelated question at 0.040**. There
is no threshold that admits one and excludes the other, and lowering the threshold to 0.08 would
admit pure noise.

The cause is structural, not a tuning error. `all-MiniLM-L6-v2` is a *symmetric* similarity model:
it measures whether two pieces of text **mean the same thing**. "What is this document about?"
is a question about the document *as an object*; it shares no meaning with sentences about sea
turtles. Content-bearing questions work exactly as designed — the 0.683 above is proof — but
meta-questions are outside what similarity search can express.

The fix is to change *retrieval mode*, not the threshold. When (a) the similarity pass returns
nothing and (b) `looks_like_overview_request()` recognises a meta-question, the retriever issues
a second, **metadata-filtered** query for `chunk_index < 4` — the opening chunks of each indexed
document — and returns them in reading order. Selection is done by the metadata filter, not by
score, so the result is deterministic and every chunk is still real, citable text from the corpus.

Two properties keep this from becoming a hallucination hole:

* It only fires when normal retrieval found **nothing**, so specific questions are untouched.
* It only fires for recognised meta-phrasings. *"What is the capital of France?"* does not match,
  so it is still refused — this is asserted directly by `test_unanswerable_question_is_still_refused`.

Confidence is scored differently for these results (§5.4), and both the UI and the JSONL log mark
the answer as overview-derived rather than similarity-matched.

### 5.4 Confidence score

```
confidence = 0.55 × best_similarity
           + 0.25 × mean_similarity
           + 0.20 × fraction_of_retrieved_sources_cited
```

The weighting reflects what each term measures. Best similarity dominates because a single
strongly matching chunk is the best available evidence that the corpus contains the answer. Mean
similarity rewards corroboration across several chunks. The citation fraction is the only term
that inspects the *generated* text, and it distinguishes an answer genuinely built from the
context from one that merely had good context available. The result is bucketed as High (≥ 0.70),
Medium (≥ 0.50) or Low, and displayed alongside the raw percentage.

Overview-fallback answers (§5.3) are scored on a separate scale — `0.45 + 0.20 × cited_fraction`
— because their context was chosen by metadata, not by similarity, so the similarity terms carry
no information. Scoring them on the normal formula would report ~5 % confidence for a correct,
fully-cited document summary. The replacement scale is deliberately capped below the "High" band:
a summary is inherently less pinpointed than a matched fact.

This is a heuristic, not a calibrated probability. It is intended to rank answers within one
corpus, and should not be read as "70 % likely to be correct".

---

## 6. Challenges faced

**1. Silent chunk degradation.** The two overlap and punctuation bugs in §2.3 produced no error
and no visible symptom — only slightly worse retrieval. They were found by printing actual chunk
text from a hand-built two-page PDF rather than by trusting the unit tests, which had been written
against synthetic strings that happened to avoid both edge cases. The lesson applied to the rest
of the project: inspect real intermediate output, and only then write the regression test.

**2. Chunk overlap poisoning top-k.** Early retrieval returned three chunks that were ~90 % the
same text, wasting the context window. Overlap and retrieval diversity pull in opposite
directions; the shingle de-duplication pass in §5.1 resolves the conflict, and over-fetching
ensures it cannot starve the result set.

**3. Dimension mismatch on model change.** Switching the embedding model against an existing index
failed deep inside the upsert loop with an unhelpful error. The startup dimension guard in §4.1
turns this into an immediate, explanatory failure.

**4. Eventual consistency.** Vectors are not queryable the instant an upsert returns, which makes
"index then immediately ask" look broken. The UI states this explicitly when a namespace reports
zero vectors.

**5. Python 3.14 dependency risk.** The development machine had only Python 3.14, a version new
enough that wheels for the deep-learning stack were not guaranteed. This motivated the pluggable
embedding backend in §3 rather than a hard dependency on local `torch`. In the event the risk did
not materialise — `torch 2.13.0` and `sentence-transformers` installed and ran correctly on 3.14 —
but the abstraction is cheap, and it remains the documented fallback for environments where the
install does fail.

**6. Pinecone SDK API drift.** The client that resolved was `pinecone 9.1.0`, several majors newer
than most published RAG tutorials, which target the v3–v5 API. Three differences matter:
`Index.query`, `Index.upsert` and `Index.delete` are now **keyword-only**; the exception type is
`PineconeError`, re-exported as `PineconeException`; and query results are `msgspec` structs
(`ScoredVector`), not dicts. Rather than pin to one major, `vector_store.py` calls everything by
keyword and normalises responses through `_match_to_dict`, which accepts a dict, an object with
`to_dict()`, or a plain attribute-bearing object. This was verified against the real SDK types and
is covered by `test_match_normalisation_accepts_every_sdk_shape`.

**7. Scanned PDFs.** A scanned document extracts to zero characters and would otherwise index an
empty corpus. This is now handled by the automatic OCR stage (§2.5); pages that individually fail
extraction are skipped and reported as a warning rather than aborting the document, and when the
OCR toolchain is absent the user gets per-OS installation guidance instead of a stack trace.

**8. A question the retriever structurally could not answer.** The most instructive failure of
the project: *"What is this document about?"* — the single most natural question a user asks —
scored 0.082 against its own document, statistically indistinguishable from an unrelated
question at 0.040. It was found only by running the finished system end to end and asking a
human-realistic question, not by any unit test. The lesson is that a retrieval threshold
validated on *content* queries says nothing about *meta* queries, and that the two need different
retrieval modes rather than different thresholds. Full analysis in §5.3.

**9. Verification without the OCR binaries.** Tesseract and Poppler are system installs, so the
Tesseract call itself could not be exercised on the development machine. Rather than leave the
feature unverified, the OCR call was stubbed at its narrowest boundary — a single function
returning what Tesseract would return — leaving *every* other stage real: a genuine image-only
PDF (confirmed to extract zero characters), real detection, real merging, real chunking, real
embeddings, a real Pinecone upsert and query, and a real Groq completion. That isolates the
unverified surface to one function call rather than to the whole feature.

---

## 7. Performance analysis

### 7.1 Measured — local pipeline stages

Measured on the development machine (Windows 11, Python 3.14.6, CPU only — no GPU), using a
hand-built two-page text PDF and a synthetic corpus of 64 chunks of 800 characters each:

| Stage | Measurement |
|---|---|
| Embedding model load (first run, includes ~80 MB download) | 47.1 s |
| Embedding model load (subsequent runs, cached) | ~2 s |
| Document embedding throughput | **74.9 chunks/s — 13.3 ms per 800-char chunk** |
| Query embedding | **14.5 ms** |
| Extraction + cleaning, 2-page PDF | < 0.05 s |
| Unit test suite (33 tests) | 0.28 s |
| Output vector shape / dtype | `(64, 384)` `float32` |
| L2 norm of every output vector | 1.0 (max deviation 1.2 × 10⁻⁷) |

Practical consequence: a 40-page document yielding roughly 250 chunks embeds locally in about
3.5 seconds. Indexing cost is therefore dominated by the Pinecone upsert round-trips, not by
embedding — which is what justifies choosing a local CPU model over a hosted embedding API.

### 7.1.1 Measured — retrieval discrimination

The similarity threshold is only meaningful if relevant and irrelevant text separate cleanly.
Measured with the actual configured model, for the query *"How does Pinecone handle metadata
filtering?"*:

| Candidate passage | Cosine similarity |
|---|---|
| "Pinecone supports metadata filtering on vector queries." | **0.7749** |
| "The cat sat on a warm windowsill during the afternoon." | **0.0304** |

The gap is roughly 25×. This is the empirical basis for the 0.35 default threshold: it sits far
above the noise floor observed for unrelated text and far below the score of a genuine match, so
the retrieval gate in §5.2 rejects irrelevant context reliably rather than by luck.

### 7.2 Chunk-size trade-off

Chunk size is exposed in the UI because the optimum is corpus-dependent, but the direction of the
trade-off is general:

| Chunk size | Retrieval precision | Context completeness | Vectors per document | Notes |
|---|---|---|---|---|
| 200–400 | High | Low | High | Good for fact lookup; answers spanning a paragraph get fragmented |
| **600–1000** | **Balanced** | **Balanced** | **Moderate** | **Recommended default (800)** |
| 1200–2000 | Lower | High | Low | Exceeds the embedder's 256-token window — text is truncated |

The 800-character default sits below the truncation limit while still holding two to four complete
sentences, which is roughly the span of a self-contained factual statement in technical prose.

### 7.3 Threshold behaviour

Consistent with the measurements in §7.1.1, cosine scores with normalised MiniLM embeddings fall
into usable bands: a direct paraphrase of document text scores above ~0.6, a topically related
passage lands around ~0.35–0.5, and unrelated text sits near ~0.03. The 0.35 default therefore
sits at the boundary between "related" and "unrelated". Raising it toward 0.5 makes the system
refuse more often but grounds the answers it does give more tightly; lowering it toward 0.25
increases recall on loosely-worded questions.

### 7.4 Not measured here — end-to-end latency and accuracy

The figures above cover every stage that runs locally. **End-to-end query latency and retrieval
accuracy were not measured for this report**, because both require live Pinecone and Groq
credentials, and both depend on the reader's own region, network and API quota. They must be
measured locally rather than copied from a report. Every query is already logged to
`logs/queries.jsonl` with `latency_seconds`, `best_score`, `chunks_retrieved`, `confidence` and
`grounded`, which makes the analysis a short pandas script:

```python
import pandas as pd
df = pd.read_json("logs/queries.jsonl", lines=True)
print(df[["latency_seconds", "best_score", "confidence"]].describe())
print("refusal rate:", 1 - df["grounded"].mean())
```

The evaluation protocol used for this report:

1. Index a representative PDF (10–40 pages).
2. Ask ten questions whose answers are present in the document, and five whose answers are not.
3. Record, from the log: the refusal rate on the answerable set (should be near zero), the refusal
   rate on the unanswerable set (**should be 100 % — this is the hallucination-prevention
   measurement**), the mean best similarity, and median end-to-end latency.
4. Repeat at chunk sizes 400 / 800 / 1200 to confirm the trade-off in §7.2 on your own corpus.

Expected latency is dominated by two network round-trips (Pinecone query, Groq completion); local
embedding of a single query on CPU is a small fraction of the total.

---

## 8. Requirement coverage summary

All six functional requirements, all four non-functional requirements, and **all seven** optional
intermediate enhancements are implemented — multi-document support, session query history,
adjustable chunk size, adjustable top-k, metadata filtering by document and page range, confidence
scoring, and persistent query logging. Section 5 of the `README.md` maps each requirement to the
specific function that satisfies it.

---

## 9. Limitations and future work

* **OCR quality is only as good as the scan.** Skewed, low-DPI or noisy pages degrade badly.
  Deskew/denoise preprocessing before recognition would help, as would surfacing Tesseract's
  per-word confidence so low-confidence pages could be flagged in the UI rather than silently
  indexed.
* **OCR is not verified against the real Tesseract binary here** (see §6.9); the call is stubbed
  at its narrowest boundary and everything around it is real.
* **The overview fallback is phrase-matched, not learned.** `looks_like_overview_request` covers
  the common English phrasings; an unusual one ("give me the elevator pitch") would fall through
  to a refusal. A small intent classifier, or an asymmetric retrieval model, would generalise.
* **Dense retrieval only.** A hybrid dense + BM25 search would improve recall on exact identifiers,
  acronyms and numbers, which dense embeddings match poorly.
* **No re-ranking.** A cross-encoder re-ranker over the over-fetched candidate set would raise
  precision at small top-k, at the cost of one more model.
* **Chunk size above ~1,200 characters silently truncates** at the embedder's token limit; the UI
  should warn rather than rely on this report.
* **Confidence is heuristic, not calibrated.** Calibrating it against human-labelled correctness
  on a held-out question set would make the percentage meaningful in absolute terms.
* **No conversational follow-up.** Query history is displayed but not fed back into retrieval, so
  "what about the second one?" will not resolve against the previous turn.
