from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List PDEBench download entries and print reproducible download commands."
    )
    parser.add_argument(
        "--pdebench-root",
        type=str,
        default=str(PROJECT_ROOT / "PDEBench_code" / "PDEBench-main"),
        help="Path to the PDEBench repository root.",
    )
    parser.add_argument(
        "--pde-name",
        action="append",
        default=[],
        help="PDE name to filter, e.g. 2d_cfd. Can be passed multiple times.",
    )
    parser.add_argument(
        "--filename-contains",
        type=str,
        default=None,
        help="Optional substring filter for PDEBench filenames.",
    )
    parser.add_argument(
        "--root-folder",
        type=str,
        default="./data/external/pdebench",
        help="Destination root folder used in the printed download command.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download matching files with urllib and verify MD5 when available.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist at the destination path.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_rows(
    rows: list[dict[str, str]],
    pde_names: list[str],
    filename_contains: str | None,
) -> list[dict[str, str]]:
    normalized_names = {name.lower() for name in pde_names}
    filtered = []
    for row in rows:
        pde = row["PDE"].lower()
        filename = row["Filename"]
        if normalized_names and pde not in normalized_names:
            continue
        if filename_contains and filename_contains not in filename:
            continue
        filtered.append(row)
    return filtered


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_row(row: dict[str, str], root_folder: Path, skip_existing: bool) -> Path:
    destination_dir = root_folder / row["Path"]
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / row["Filename"]
    if destination.exists() and skip_existing:
        print(f"Skip existing file: {destination}")
        return destination

    print(f"Downloading {row['URL']} -> {destination}")
    with urllib.request.urlopen(row["URL"]) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    expected_md5 = row.get("MD5", "").strip().lower()
    if expected_md5:
        actual_md5 = md5_file(destination)
        if actual_md5.lower() != expected_md5:
            raise RuntimeError(
                f"MD5 mismatch for {destination}: expected {expected_md5}, got {actual_md5}"
            )
    return destination


def main() -> None:
    args = parse_args()
    pdebench_root = Path(args.pdebench_root).expanduser().resolve()
    csv_path = pdebench_root / "pdebench" / "data_download" / "pdebench_data_urls.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PDEBench URL CSV not found: {csv_path}. "
            "Download or clone PDEBench under PDEBench_code/PDEBench-main first."
        )

    rows = filter_rows(load_rows(csv_path), args.pde_name, args.filename_contains)
    if not rows:
        print("No matching PDEBench files found.", file=sys.stderr)
        return

    print(f"Matched {len(rows)} PDEBench file(s):")
    for row in rows:
        destination = Path(args.root_folder) / row["Path"] / row["Filename"]
        print(f"- PDE={row['PDE']} filename={row['Filename']}")
        print(f"  URL={row['URL']}")
        print(f"  destination={destination}")

    if args.download:
        print()
        root_folder = Path(args.root_folder).expanduser()
        for row in rows:
            download_row(row, root_folder=root_folder, skip_existing=args.skip_existing)
        return

    unique_pdes = sorted({row["PDE"].lower() for row in rows})
    print()
    print("Download command(s):")
    data_download_dir = pdebench_root / "pdebench" / "data_download"
    for pde_name in unique_pdes:
        print(
            f"cd {data_download_dir} && python download_direct.py "
            f"--root_folder {args.root_folder} --pde_name {pde_name}"
        )


if __name__ == "__main__":
    main()
