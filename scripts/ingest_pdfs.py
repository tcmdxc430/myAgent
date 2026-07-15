import argparse
from pathlib import Path

from ingestion.pdf import PdfImporter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import local PDF files into the PostgreSQL + Chroma knowledge base."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="PDF files to import")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    importer = PdfImporter()
    failed = False
    for path in args.paths:
        result = importer.import_path(path, force_refresh=args.force_refresh)
        print(
            f"{path.name}\tstatus={result.status}\tchunks={result.chunk_count}"
            f"\tarticle_id={result.article_id or '-'}\ttitle={result.title or '-'}"
        )
        if result.status == "failed":
            failed = True
            print(f"  error: {result.message}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
