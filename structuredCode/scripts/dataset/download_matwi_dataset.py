import argparse
from pathlib import Path
from dataset.download import (
    get_file_list_from_api,
    download_file,
    extract_zip,
)
from structuredCode.constants import MATWI_DATA_PATH

BASE_URL      = "https://rdr.kuleuven.be"
PERSISTENT_ID = "doi:10.48804/GK6LHH"
API_FILES_URL = f"{BASE_URL}/api/datasets/:persistentId/versions/:latest/files?persistentId={PERSISTENT_ID}"

KNOWN_FILES = (
    ["labels.csv", None],
    ["README.md",  None],
    ["sets.csv", None],
    *[[f"Set{i}.zip", None] for i in range(1, 18)],
)

def build_download_url(file_id: int) -> str:
    return f"{BASE_URL}/api/access/datafile/{file_id}"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the MATWI dataset.")
    p.add_argument("--output_dir", type=str, default=MATWI_DATA_PATH,
                   help=f"Root directory to save the dataset (default: {MATWI_DATA_PATH})")
    p.add_argument("--sets", nargs="*", type=int, default=None,
                   help="Which Set numbers to download, e.g. --sets 1 2 3. "
                        "Omit to download all 17.")
    p.add_argument("--resume", action="store_true",
                   help="Resume partially downloaded files.")
    p.add_argument("--no_extract", action="store_true",
                   help="Keep zip files without extracting them.")
    p.add_argument("--keep_zips", action="store_true",
                   help="Keep zip files after extraction (default: delete them).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_files = get_file_list_from_api(API_FILES_URL)

    download_map: dict[str, str] = {}

    if api_files:
        for entry in api_files:
            try:
                fname = entry["dataFile"]["filename"]
                fid   = entry["dataFile"]["id"]
                download_map[fname] = build_download_url(fid)
            except KeyError:
                continue
    else:
        for fname, _ in KNOWN_FILES:
            url = (
                f"{BASE_URL}/api/access/datafile/:persistentId"
                f"?persistentId={PERSISTENT_ID}&filename={fname}"
            )
            download_map[fname] = url

    files_to_download: list[tuple[str, str]] = []

    for fname, url in download_map.items():
        if fname in ("labels.csv", "README.md", "sets.csv"):
            files_to_download.append((fname, url))
        elif fname.startswith("Set") and fname.endswith(".zip"):
            set_num = int(fname.replace("Set", "").replace(".zip", ""))
            if args.sets is None or set_num in args.sets:
                files_to_download.append((fname, url))

    def sort_key(item):
        name = item[0]
        if name == "README.md":   return (0, 0)
        if name == "labels.csv":  return (1, 0)
        if name == "sets.csv": return (2, 0)
        try:
            return (3, int(name.replace("Set","").replace(".zip","")))
        except ValueError:
            return (4, 0)

    files_to_download.sort(key=sort_key)

    print(f"\n Will download {len(files_to_download)} file(s) to folder: {output_dir}\n")

    # ── 3. Download ────────────────────────────────────────────────────────────
    failed: list[str] = []
    zips_to_extract: list[Path] = []

    for fname, url in files_to_download:
        dest = output_dir / fname
        print(f" {fname}")

        if args.resume and dest.exists() and not fname.endswith(".zip"):
            print(f"  SKIP {fname} already exists.")
            continue

        ok = download_file(url, dest, resume=args.resume)

        if not ok:
            failed.append(fname)
            continue

        if fname.endswith(".zip") and not args.no_extract:
            zips_to_extract.append(dest)

    # ── 4. Extract zips ────────────────────────────────────────────────────────
    if zips_to_extract:
        print(f"\n Extracting {len(zips_to_extract)} zip archive(s)…\n")
        for zip_path in zips_to_extract:
            extract_zip(zip_path, output_dir)
            if not args.keep_zips:
                zip_path.unlink()
                print(f"DELETE {zip_path.name}")

    # ── 5. Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failed:
        print(f" {len(failed)} file(s) failed to download:")
        for f in failed:
            print(f"  • {f}")
        print("\nre-run with --resume to retry failed files.")
    else:
        print(" All files downloaded successfully.")

    print(f"\nDataset location: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()