# Research Paper OCR & Deep Metadata Extraction

**Project focus:** FrontierAtlas.co  
**Role:** AI / Data Engineering Intern

This project downloads five selected arXiv papers, extracts their PDF text, retrieves core bibliographic metadata from the arXiv API, and saves structured records to `extracted_papers.json`.

## Papers

- `2503.18382` — PP-FormulaNet
- `2510.13998` — BitNet Distillation
- `2607.18859` — PhoenixRepair
- `2507.23370` — Trae Agent
- `2503.11651` — VGGT: Visual Geometry Grounded Transformer

## Features

- Downloads PDFs from arXiv and reuses existing PDFs larger than 10 KB.
- Extracts selectable PDF text with PyMuPDF.
- Cleans common line-break and whitespace artifacts.
- Retrieves title, authors, abstract, and original arXiv submission date from the arXiv API.
- Saves one cleaned text file per paper in `extracted_text/`.
- Writes structured metadata to `extracted_papers.json`.
- Includes JSON structure validation and an offline self-test.
- Attempts OCR only on pages with very little extractable text, if optional OCR dependencies and the Tesseract application are installed.

## Requirements

- Python 3.10 or newer
- Internet connection for downloading PDFs and accessing the arXiv API
- Tesseract OCR application only if you want OCR fallback for scanned/image-only pages

## Setup on Windows

Open Command Prompt and enter your project folder. For your current setup:

```bat
cd "%USERPROFILE%\Desktop\research_paper_extractor"
```

Create and activate the virtual environment (skip creation if `.venv` already exists):

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Optional OCR setup

The Python packages `pytesseract` and `Pillow` are installed from `requirements.txt`, but OCR also requires the separate Tesseract application. Install Tesseract for Windows and make sure the `tesseract` command is available in Command Prompt. If Tesseract is not installed or cannot be found, the script continues with normal text extraction and prints a note.

## Run the extractor

```bat
python extract_paper_meta.py
```

To run offline structural checks against the existing JSON:

```bat
python extract_paper_meta.py --self-test
```

To explicitly disable OCR attempts:

```bat
python extract_paper_meta.py --no-ocr
```

## Output structure

```text
research_paper_extractor/
├── .venv/
├── papers/                  # downloaded PDF files
├── extracted_text/          # cleaned text, one file per paper
├── extract_paper_meta.py
├── requirements.txt
├── extracted_papers.json
└── README.md
```

Each paper record includes:
- `arxiv_id`, `title`, `authors`, `abstract`
- `publication_date` in `YYYY-MM-DD` format
- `arxiv_url`, `github_url`
- `models`, `datasets`
- `sota_results` with benchmark, metric, value, comparison target, and context
- PDF page count and text-extraction/OCR diagnostics

## How metadata is sourced

- **Bibliographic fields:** fetched from the arXiv API.
- **Models and datasets:** paper-specific seed lists are supplemented by known candidate terms only when those terms appear in the extracted paper text. This is not a universal named-entity recognition model, so results require review.
- **Official repositories:** curated links intended to point to the repository associated with the paper; verify each against the paper or its arXiv page.
- **SOTA claims:** curated abstract-level claims. The script does not claim to reliably infer table cell alignment or automatically discover every benchmark result. Verify numbers and comparison targets against the original paper before downstream use.
- **OCR:** only attempted for pages with very little text, and only when both Python OCR dependencies and the separate Tesseract executable are available.

## Validation checklist

- [ ] Run the script and confirm all five papers process.
- [ ] Run `python extract_paper_meta.py --self-test`.
- [ ] Confirm each paper's title, authors, abstract, and submission date against arXiv.
- [ ] Confirm repository URLs are officially associated with each paper.
- [ ] Review dataset and model names against the full extracted text.
- [ ] Verify every benchmark claim against the original paper's abstract or table.
- [ ] If OCR is required, test with an image-only PDF page and confirm OCR is actually used.

## Limitations

This is an educational extraction pipeline for five selected papers, not a fully general scientific-paper understanding system. Text extraction success does not guarantee complete metadata coverage. Curated model, dataset, repository, and result fields may need manual review. OCR fallback is optional and depends on a separate Tesseract installation.
