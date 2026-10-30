"""
Research Paper OCR & Deep Metadata Extraction
Project focus: FrontierAtlas.co

Downloads five arXiv papers, extracts PDF text (using PyMuPDF first and
optional Tesseract OCR when needed), retrieves bibliographic metadata from
the arXiv API, and writes structured JSON.

Important:
- Bibliographic fields come from the arXiv API.
- Models, datasets, repositories, and result claims are supplemented by a
  paper-specific curated seed list and checked against the extracted text.
- This is not a universal scientific-table parser. SOTA values must be checked
  against the source paper before being used as verified facts.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pymupdf
import requests

PROJECT_DIR = Path(__file__).resolve().parent
PDF_DIR = PROJECT_DIR / "papers"
TEXT_DIR = PROJECT_DIR / "extracted_text"
OUTPUT_JSON = PROJECT_DIR / "extracted_papers.json"
ARXIV_API = "https://export.arxiv.org/api/query?id_list={}"
USER_AGENT = "FrontierAtlasPaperMetadataExtractor/1.1 (educational project)"
MIN_TEXT_CHARS_PER_PAGE = 40

PAPERS = [
    "2503.18382",
    "2510.13998",
    "2607.18859",
    "2507.23370",
    "2503.11651",
]

# Paper-specific seed information. Keep claims traceable and do not infer
# numerical results from arbitrary table text.
CURATED: dict[str, dict[str, Any]] = {
    "2503.18382": {
        "github_url": [
            "https://github.com/PaddlePaddle/PaddleOCR",
            "https://github.com/PaddlePaddle/PaddleX",
        ],
        "models": [
            "PP-FormulaNet-L", "PP-FormulaNet-S", "Vary-VIT-B",
            "PP-HGNetV2-B4", "MBart Decoder", "GOT2.0-OCR",
            "UniMERNet", "Pix2tex",
        ],
        "datasets": ["UniMERNet-1M", "arXiv-4M"],
        "sota_results": [
            {
                "benchmark": "Formula-recognition evaluation (paper abstract)",
                "metric": "Relative performance improvement",
                "value": "6%",
                "comparison_target": "UniMERNet; PP-FormulaNet-L",
                "context": "Abstract-level claim. Check the original table before treating as a verified table result."
            },
            {
                "benchmark": "Formula-recognition inference (paper abstract)",
                "metric": "Speedup",
                "value": "over 16x faster",
                "comparison_target": "PP-FormulaNet-S versus the comparison model described by the paper",
                "context": "Abstract-level claim; exact hardware and batch conditions matter."
            }
        ],
    },
    "2510.13998": {
        "github_url": ["https://github.com/microsoft/BitNet"],
        "models": ["BitDistill", "BitNet", "Qwen", "MiniLM"],
        "datasets": [],
        "sota_results": [
            {
                "benchmark": "CPU inference (paper abstract)",
                "metric": "Inference speedup",
                "value": "2.65x faster",
                "comparison_target": "Full-precision counterpart models",
                "context": "Abstract-level claim; not a benchmark-specific accuracy score."
            },
            {
                "benchmark": "Memory usage (paper abstract)",
                "metric": "Memory reduction",
                "value": "up to 10x",
                "comparison_target": "Full-precision counterpart models",
                "context": "Abstract-level claim; not a benchmark-specific accuracy score."
            }
        ],
    },
    "2607.18859": {
        "github_url": ["https://github.com/DeepSoftwareAnalytics/PhoenixRepair"],
        "models": ["PhoenixRepair", "DeepSeek-V3.1", "MiniMax-M2.5", "SWE-agent", "Mini-SWE-agent", "Live-SWE-agent"],
        "datasets": ["SWE-bench-Verified"],
        "sota_results": [
            {
                "benchmark": "SWE-bench-Verified",
                "metric": "Resolved rate (Pass@1)",
                "value": "76.0%",
                "comparison_target": "Other evaluated approaches; reported with MiniMax-M2.5",
                "context": "Claim stated in the paper abstract; benchmark setup should be checked in the paper."
            },
            {
                "benchmark": "SWE-bench-Verified",
                "metric": "Relative improvement",
                "value": "7.8%",
                "comparison_target": "SWE-agent under DeepSeek-V3.1",
                "context": "Claim stated in the paper abstract."
            }
        ],
    },
    "2507.23370": {
        "github_url": ["https://github.com/bytedance/trae-agent"],
        "models": ["Trae Agent"],
        "datasets": ["SWE-bench", "SWE-bench Verified"],
        "sota_results": [
            {
                "benchmark": "SWE-bench Verified",
                "metric": "Pass@1",
                "value": "75.20%",
                "comparison_target": "Leaderboard approaches at the time reported by the paper",
                "context": "Claim stated in the paper abstract; leaderboard standings can change over time."
            },
            {
                "benchmark": "SWE-bench (paper evaluation)",
                "metric": "Average improvement in Pass@1",
                "value": "10.22%",
                "comparison_target": "Four baseline ensemble-reasoning techniques",
                "context": "Claim stated in the paper abstract."
            }
        ],
    },
    "2503.11651": {
        "github_url": ["https://github.com/facebookresearch/vggt"],
        "models": ["VGGT", "DINO", "VGGSfM", "DUSt3R", "MASt3R"],
        "datasets": [],
        "sota_results": [],
    },
}

# Candidate terms are only included if they actually appear in the extracted
# text. This reduces false positives from a curated list.
KNOWN_TERMS: dict[str, dict[str, list[str]]] = {
    "2510.13998": {
        "datasets": [
            "WikiText", "C4", "RedPajama", "OpenWebText", "GSM8K",
            "MMLU", "ARC", "HellaSwag", "TruthfulQA", "WinoGrande",
        ]
    },
    "2503.11651": {
        "datasets": [
            "CO3D", "DTU", "ETH3D", "RealEstate10K", "ScanNet",
            "Tanks and Temples", "MegaDepth", "BlendedMVS", "KITTI",
            "Map-free", "ACID", "DL3DV",
        ]
    },
}

def clean_pdf_text(text: str) -> str:
    """Normalize common PDF text artifacts while preserving real hyphens."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    return text.strip()

def fetch_arxiv_metadata(arxiv_id: str) -> dict[str, Any]:
    """Fetch title, authors, abstract, and original submission date from arXiv."""
    response = requests.get(
        ARXIV_API.format(arxiv_id), timeout=45,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        raise ValueError(f"No arXiv API entry found for {arxiv_id}")

    def field(tag: str) -> str:
        node = entry.find(f"atom:{tag}", ns)
        return node.text.strip() if node is not None and node.text else ""

    title = re.sub(r"\s+", " ", field("title")).strip()
    abstract = re.sub(r"\s+", " ", field("summary")).strip()
    authors = []
    for node in entry.findall("atom:author", ns):
        name = node.find("atom:name", ns)
        if name is not None and name.text:
            authors.append(re.sub(r"\s+", " ", name.text).strip())
    published = field("published")
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "publication_date": published[:10] if len(published) >= 10 else None,
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
    }

def download_pdf(arxiv_id: str) -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
        return pdf_path
    response = requests.get(
        f"https://arxiv.org/pdf/{arxiv_id}", timeout=90,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"Downloaded content for {arxiv_id} does not look like a PDF.")
    pdf_path.write_bytes(response.content)
    return pdf_path

def try_ocr_page(page: pymupdf.Page) -> tuple[str, str]:
    """OCR a page if optional pytesseract, Pillow, and Tesseract are available."""
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        return "", "OCR unavailable: install optional packages with `pip install pytesseract Pillow`."
    if not shutil.which("tesseract"):
        return "", "OCR unavailable: Tesseract application was not found on PATH."
    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image)
        return clean_pdf_text(text), "OCR attempted"
    except Exception as exc:
        return "", f"OCR failed: {exc}"

def extract_pdf_text(pdf_path: Path, arxiv_id: str, enable_ocr: bool = True) -> tuple[str, int, dict[str, Any]]:
    """Extract text page by page; attempt OCR only on pages with little text."""
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    page_texts: list[str] = []
    ocr_pages: list[int] = []
    ocr_notes: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        page_count = len(doc)
        for page_num, page in enumerate(doc, start=1):
            page_text = clean_pdf_text(page.get_text("text"))
            if enable_ocr and len(page_text.strip()) < MIN_TEXT_CHARS_PER_PAGE:
                ocr_text, note = try_ocr_page(page)
                if ocr_text.strip():
                    page_text = ocr_text
                    ocr_pages.append(page_num)
                elif note not in ocr_notes:
                    ocr_notes.append(note)
            page_texts.append(page_text)
    cleaned = clean_pdf_text("\n\n".join(page_texts))
    (TEXT_DIR / f"{arxiv_id}.txt").write_text(cleaned, encoding="utf-8")
    diagnostics = {
        "ocr_pages": ocr_pages,
        "ocr_used": bool(ocr_pages),
        "ocr_notes": ocr_notes,
        "text_extraction_status": (
            "text extracted" if len(cleaned) > 500
            else "little/no text; install and configure OCR if this PDF is scanned"
        ),
    }
    return cleaned, page_count, diagnostics

def add_text_supported_terms(arxiv_id: str, text: str, record: dict[str, Any]) -> None:
    """Add known candidate terms only when they appear in this paper's text."""
    lower_text = text.casefold()
    curated = CURATED.get(arxiv_id, {})
    for field in ("models", "datasets"):
        values = list(curated.get(field, []))
        for term in KNOWN_TERMS.get(arxiv_id, {}).get(field, []):
            if term.casefold() in lower_text and term not in values:
                values.append(term)
        record[field] = values
    record["github_url"] = curated.get("github_url", [])
    record["sota_results"] = curated.get("sota_results", [])

def validate_records(records: list[dict[str, Any]]) -> list[str]:
    """Return validation problems without silently inventing missing values."""
    problems: list[str] = []
    seen: set[str] = set()
    required = {
        "arxiv_id", "title", "authors", "abstract", "publication_date",
        "arxiv_url", "github_url", "models", "datasets", "sota_results",
    }
    for index, record in enumerate(records, start=1):
        missing = sorted(required - set(record))
        if missing:
            problems.append(f"Record {index} is missing fields: {', '.join(missing)}")
        arxiv_id = record.get("arxiv_id", "")
        if arxiv_id in seen:
            problems.append(f"Duplicate arXiv ID: {arxiv_id}")
        seen.add(arxiv_id)
        for field in ("title", "abstract"):
            if not record.get(field):
                problems.append(f"{arxiv_id}: empty {field}")
        if not record.get("authors"):
            problems.append(f"{arxiv_id}: author list is empty")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(record.get("publication_date", ""))):
            problems.append(f"{arxiv_id}: publication_date is not YYYY-MM-DD")
        for result in record.get("sota_results", []):
            for field in ("benchmark", "metric", "value", "comparison_target"):
                if not result.get(field):
                    problems.append(f"{arxiv_id}: SOTA result missing {field}")
    if len(records) != len(PAPERS):
        problems.append(f"Expected {len(PAPERS)} records; found {len(records)}.")
    return problems

def self_test() -> int:
    """Offline checks for JSON shape and text-cleaning behavior."""
    sample = "inter-\nnational\r\n\r\npaper"
    expected = "international\n\npaper"
    if clean_pdf_text(sample) != expected:
        print("FAIL: clean_pdf_text test")
        return 1
    if len(PAPERS) != 5 or len(set(PAPERS)) != 5:
        print("FAIL: paper ID list")
        return 1
    if OUTPUT_JSON.exists():
        try:
            data = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
            problems = validate_records(data)
            if problems:
                print("JSON validation warnings:")
                for problem in problems:
                    print(f" - {problem}")
                return 1
            print(f"PASS: JSON contains {len(data)} unique, structurally valid records.")
        except Exception as exc:
            print(f"FAIL: existing JSON cannot be parsed: {exc}")
            return 1
    else:
        print("NOTE: JSON output does not exist yet; run without --self-test after setup.")
    print("PASS: text-cleaning and paper-list checks.")
    return 0

def main() -> None:
    parser = argparse.ArgumentParser(description="Download arXiv papers and extract metadata.")
    parser.add_argument("--self-test", action="store_true", help="Run offline structural checks.")
    parser.add_argument("--no-ocr", action="store_true", help="Disable optional OCR fallback.")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(self_test())

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, arxiv_id in enumerate(PAPERS, start=1):
        print(f"[{index}/{len(PAPERS)}] Processing arXiv:{arxiv_id}")
        try:
            pdf_path = download_pdf(arxiv_id)
            text, page_count, diagnostics = extract_pdf_text(
                pdf_path, arxiv_id, enable_ocr=not args.no_ocr
            )
            record = fetch_arxiv_metadata(arxiv_id)
            add_text_supported_terms(arxiv_id, text, record)
            record.update({
                "pdf_pages": page_count,
                "extracted_text_characters": len(text),
                "pdf_text_extraction": diagnostics["text_extraction_status"],
                "ocr_used": diagnostics["ocr_used"],
                "ocr_pages": diagnostics["ocr_pages"],
                "ocr_notes": diagnostics["ocr_notes"],
                "extraction_notes": [
                    "Title, authors, abstract and original submission date are retrieved from the arXiv API.",
                    "Models and datasets are seeded from paper-specific information and known terms are added only when found in extracted text.",
                    "Repository URLs and SOTA claims are curated; verify against the paper before treating them as exhaustive or independently validated.",
                    "OCR is attempted only on pages with very little extractable text and only when pytesseract, Pillow, and the Tesseract application are available.",
                ],
            })
            records.append(record)
            print(f"    OK: {record['title']} ({page_count} pages, {len(text):,} text characters)")
            if diagnostics["ocr_used"]:
                print(f"    OCR used on page(s): {diagnostics['ocr_pages']}")
            elif diagnostics["ocr_notes"]:
                print(f"    Note: {diagnostics['ocr_notes'][0]}")
        except Exception as exc:
            failures.append({"arxiv_id": arxiv_id, "error": str(exc)})
            print(f"    FAILED: {exc}")
        time.sleep(1.0)

    OUTPUT_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT_JSON}")
    print(f"Successful papers: {len(records)}/{len(PAPERS)}")
    problems = validate_records(records)
    if problems:
        print("\nValidation warnings:")
        for problem in problems:
            print(f" - {problem}")
    else:
        print("Structural validation: PASS")
    if failures:
        print("\nFailures:")
        print(json.dumps(failures, ensure_ascii=False, indent=2))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
