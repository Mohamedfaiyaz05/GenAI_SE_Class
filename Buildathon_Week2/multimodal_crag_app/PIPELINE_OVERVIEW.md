# PDF Extraction and Vector Store

## How a PDF becomes searchable

```text
PDF in data/
  → extract text, tables, and page visuals
  → create source-labelled chunks
  → generate embeddings
  → save chunks and vectors in local FAISS index
```

1. **Discover and filter PDFs** — The extractor scans `data/`. The vector manager skips configured exclusions; `spotify_web_app_architecture.pdf` is excluded by default.
2. **Extract page content** — [src/extractor.py](./src/extractor.py) reads selectable PDF text and tables with PyMuPDF. Text is split into chunks of up to 1,200 characters with 150 characters of overlap. Pages containing an embedded image of at least 5 KB are rendered as a whole page and sent to OpenAI `gpt-4o-mini`; this keeps image captions together with the images they describe.
3. **Attach source metadata** — Each chunk records its filename, page number, section, and type (`text`, `table`, or `visual_chart`) so retrieved evidence can be traced back to a source.
4. **Embed and store** — [src/vector_store.py](./src/vector_store.py) uses OpenAI `text-embedding-3-small` to turn each chunk into a vector, then stores the vectors and original text in FAISS under `vector_index/`.
5. **Avoid accidental repeat ingestion** — `processed_files.json` records indexed filenames. Normal ingestion processes only new, eligible files. The UI separates uploading a PDF from explicitly creating its chunks and indexing them.

## Problems addressed

| Problem | What the pipeline does |
|---|---|
| A relevant fact is buried in a long page | Bounded, overlapping text chunks improve retrieval precision while retaining nearby context. |
| A caption is visible on the page but not selectable as PDF text | Full-page vision extraction can transcribe captions, labels, dates, and chart text. |
| Search terms differ from the source wording | Hybrid retrieval combines semantic search with keyword matching, close typo correction, and the `tamilnadu` → `tamil nadu` alias. |
| Irrelevant PDFs pollute agricultural search | Configured exclusions filter PDFs from ingestion and retrieval. |
| Rebuilding could leave the application without its old index if it fails | `rebuild_index` builds in a staging directory and swaps it into place only after successful ingestion. |
| A running backend continues using a replaced index | The vector manager checks the saved index file signature and reloads it when the index changes. |

## Current indexed data

The local index contains **553 chunks** from `agri_schemes_2026_27.pdf` and `budget_doc.pdf`. Page 299's image-only caption was transcribed into a searchable chunk and verified with retrieval. The unrelated Spotify PDF remains outside the index.

## Chat experience

- Answers follow the language of the question; Tamil-script questions receive Tamil responses, while citations and source names remain intact.
- Each assistant answer has compact like/dislike controls. Ratings are saved in `feedback/` by response ID; the stored record contains the rating and timestamp, not the question, answer, or session ID. A user can change their rating.

## Limitations to know

- Vision extraction is triggered by embedded PDF images of at least 5 KB. A scanned page with no detectable embedded image, or a low-quality image, may still need dedicated OCR or manual review.
- Image extraction requires a working OpenAI API connection and a valid `OPENAI_API_KEY`; API errors stop ingestion rather than saving an error message as evidence.
- The processed-file registry tracks **filenames**, not file contents. Replacing a PDF with a different file of the same name requires an explicit index rebuild.
- The local FAISS index and registry are derived data. Back them up if they must survive workspace cleanup.

## Validation

Focused offline tests cover text chunking, full-page caption extraction, typo-tolerant retrieval, index replacement, and upload/indexing behavior. A live vision extraction and retrieval check confirmed the page 299 Chief Minister caption is searchable.
