"""
scripts/ingest_docs.py — CLI for bulk-ingesting supplier documents.

Usage
-----
    # Ingest everything in a directory (auto-detect doc_type from filename):
    python scripts/ingest_docs.py data/supplier_docs/

    # Ingest a single file with explicit metadata:
    python scripts/ingest_docs.py path/to/file.pdf \\
        --supplier "Apex Supply Co." --doc-type sla

    # Ingest a directory, restrict to one supplier:
    python scripts/ingest_docs.py data/supplier_docs/ \\
        --supplier "Clearline Logistics"

    # Dry-run (show what would be ingested, skip DB writes):
    python scripts/ingest_docs.py data/supplier_docs/ --dry-run

Doc-type auto-detection
-----------------------
If --doc-type is not given, the script infers it from the filename:
    *_sla.*        → sla
    *_contract.*   → contract
    *_policy.*     → policy
    (anything else) → contract   (conservative default)

Supplier auto-detection
-----------------------
If --supplier is not given, the script infers it from the filename stem by
replacing underscores with spaces and stripping the trailing _sla / _contract
/ _policy suffix.  Example: apex_supply_co_sla.txt → "Apex Supply Co."

Supported file formats
----------------------
  .pdf   — text extracted with pypdf
  .txt   — decoded as UTF-8 (latin-1 fallback)

Exit codes
----------
  0  — all files processed successfully (or dry-run completed)
  1  — one or more files failed or errored
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.ingestor import ingest_document  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

_DOC_TYPE_KEYWORDS = {
    "sla": "sla",
    "contract": "contract",
    "policy": "policy",
}


def _infer_doc_type(stem: str) -> str:
    """
    Infer document type from the filename stem.

    Args:
        stem: Filename without extension (e.g. 'apex_supply_co_sla').

    Returns:
        One of 'sla', 'contract', 'policy', or 'contract' as default.
    """
    stem_lower = stem.lower()
    for keyword, doc_type in _DOC_TYPE_KEYWORDS.items():
        if stem_lower.endswith(f"_{keyword}") or f"_{keyword}_" in stem_lower:
            return doc_type
    return "contract"


def _infer_supplier_name(stem: str) -> str:
    """
    Infer supplier name from filename stem by stripping the doc-type suffix
    and converting underscores to title-case words.

    Args:
        stem: Filename without extension (e.g. 'apex_supply_co_sla').

    Returns:
        A human-readable supplier name string (e.g. 'Apex Supply Co').
    """
    # Strip known doc-type suffixes from the right.
    for keyword in _DOC_TYPE_KEYWORDS:
        if stem.lower().endswith(f"_{keyword}"):
            stem = stem[: -(len(keyword) + 1)]
            break

    # Replace underscores with spaces and title-case each word.
    return re.sub(r"_+", " ", stem).title()


def _collect_files(path: Path, supplier_filter: str | None) -> list[Path]:
    """
    Collect supported files from *path* (file or directory).

    Args:
        path: A file or directory path.
        supplier_filter: If given, only include files whose inferred supplier
            name starts with this string (case-insensitive).

    Returns:
        Sorted list of file paths.
    """
    if path.is_file():
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file type: %s", path.name)
            return []
        return [path]

    if not path.is_dir():
        logger.error("Path does not exist: %s", path)
        return []

    files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    if supplier_filter:
        sf_lower = supplier_filter.lower()
        files = [
            f for f in files
            if _infer_supplier_name(f.stem).lower().startswith(sf_lower)
        ]
        if not files:
            logger.warning(
                "No files matched supplier filter '%s' in %s.", supplier_filter, path
            )

    return files


def _process_file(
    file_path: Path,
    supplier_name: str | None,
    doc_type: str | None,
    dry_run: bool,
) -> tuple[str, dict]:
    """
    Process a single file: read bytes, infer metadata, and call ingest_document.

    Returns:
        (status_str, result_dict) where status_str is 'ok', 'duplicate', or 'error'.
    """
    inferred_supplier = supplier_name or _infer_supplier_name(file_path.stem)
    inferred_doc_type = doc_type or _infer_doc_type(file_path.stem)

    if dry_run:
        info = {
            "filename": file_path.name,
            "supplier_name": inferred_supplier,
            "doc_type": inferred_doc_type,
            "size_bytes": file_path.stat().st_size,
        }
        logger.info(
            "[DRY-RUN] Would ingest: %s | supplier=%s | doc_type=%s | size=%d bytes",
            file_path.name,
            inferred_supplier,
            inferred_doc_type,
            file_path.stat().st_size,
        )
        return "dry_run", info

    try:
        file_bytes = file_path.read_bytes()
    except OSError as exc:
        error_info = {"filename": file_path.name, "error": str(exc)}
        logger.error("Cannot read file '%s': %s", file_path.name, exc)
        return "error", error_info

    raw_result = ingest_document(
        file_bytes=file_bytes,
        filename=file_path.name,
        supplier_name=inferred_supplier,
        doc_type=inferred_doc_type,
    )

    try:
        result = json.loads(raw_result)
    except json.JSONDecodeError:
        result = {"status": "error", "message": f"Bad JSON from ingestor: {raw_result}"}

    status = result.get("status", "error")
    return status, result


def main() -> int:
    """
    Entry point for the ingest CLI.

    Returns:
        Exit code: 0 = all OK, 1 = any errors.
    """
    parser = argparse.ArgumentParser(
        description="Bulk-ingest supplier documents into the vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a single file or a directory of supplier documents.",
    )
    parser.add_argument(
        "--supplier",
        metavar="NAME",
        default=None,
        help="Supplier name override (inferred from filename if omitted).",
    )
    parser.add_argument(
        "--doc-type",
        metavar="TYPE",
        choices=["sla", "contract", "policy"],
        default=None,
        help="Document type override: sla | contract | policy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without writing to the database.",
    )

    args = parser.parse_args()

    files = _collect_files(args.path, args.supplier)

    if not files:
        logger.error("No supported files found at: %s", args.path)
        return 1

    logger.info(
        "%s%d file(s) to process...",
        "[DRY-RUN] " if args.dry_run else "",
        len(files),
    )

    counts: dict[str, int] = {"ok": 0, "duplicate": 0, "error": 0, "dry_run": 0}

    for file_path in files:
        status, result = _process_file(
            file_path=file_path,
            supplier_name=args.supplier,
            doc_type=args.doc_type,
            dry_run=args.dry_run,
        )
        counts[status] = counts.get(status, 0) + 1

        if status == "ok":
            logger.info(
                "  ✓ Ingested '%s' → doc_id=%s  chunks=%s  supplier='%s'  type=%s",
                result.get("filename"),
                result.get("document_id"),
                result.get("chunks_stored"),
                result.get("supplier_name"),
                result.get("doc_type"),
            )
        elif status == "duplicate":
            logger.info("  ~ Duplicate (skipped): %s", result.get("message", ""))
        elif status == "dry_run":
            pass  # already logged above
        else:
            logger.error("  ✗ Error for '%s': %s", file_path.name, result.get("message"))

    # Summary
    print("\n" + "-" * 60)
    if args.dry_run:
        print(f"DRY-RUN complete. Would process: {counts.get('dry_run', 0)} file(s).")
    else:
        print(
            f"Done.  "
            f"Ingested: {counts['ok']}  |  "
            f"Duplicates: {counts['duplicate']}  |  "
            f"Errors: {counts['error']}"
        )
    print("-" * 60)

    return 1 if counts.get("error", 0) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
