import argparse
import logging
from pathlib import Path

from ase.db import connect

from al_dirac.logging_utils import setup_logger
from al_dirac.parser.structure_parser import find_and_store_structures

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="al-dirac")

    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-structures")
    import_parser.add_argument("--base-dir", type=Path, required=True)
    import_parser.add_argument("--db-path", type=Path, required=True)
    import_parser.add_argument("--filename-glob", default="*")
    import_parser.add_argument("--file-format", default=None)
    import_parser.add_argument("--read-index", default=":")
    import_parser.add_argument("--split", default="pool")
    import_parser.add_argument("--iteration", type=int, default=0)
    import_parser.add_argument(
        "--is-labeled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override automatic energy-and-forces label detection.",
    )
    import_parser.add_argument("--is-selected", action="store_true")
    import_parser.add_argument("--source", default=None)
    import_parser.add_argument("--keyword-in-filename", default=None)
    import_parser.add_argument("--keyword-in-file", default=None)
    import_parser.add_argument("--require-sibling-filename", default=None)
    import_parser.add_argument("--non-recursive", action="store_true")
    import_parser.add_argument("--raise-on-read-error", action="store_true")
    
    parser.add_argument("--log-level", default="INFO")

    return parser

def run_import_structures(args: argparse.Namespace) -> int:
    db = connect(str(args.db_path))

    row_ids = find_and_store_structures(
        base_dir=args.base_dir,
        db=db,
        filename_glob=args.filename_glob,
        recursive=not args.non_recursive,
        file_format=args.file_format,
        read_index=args.read_index,
        split=args.split,
        iteration=args.iteration,
        is_labeled=args.is_labeled,
        is_selected=args.is_selected,
        source=args.source,
        keyword_in_filename=args.keyword_in_filename,
        keyword_in_file=args.keyword_in_file,
        require_sibling_filename=args.require_sibling_filename,
        raise_on_read_error=args.raise_on_read_error,
    )
    
    print(f"Imported {len(row_ids)} structures into {args.db_path}")
    return 0

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    setup_logger(level=level)

    if args.command == "import-structures":
        return run_import_structures(args)

    raise ValueError(f"Unsupported command: {args.command}")

if __name__ == "__main__":
    raise SystemExit(main())
    
