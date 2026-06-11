import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

from constants.matwi_dataset_constants import API_FILES_URL, BASE_URL, CHUNK_SIZE

def get_file_list_from_api() -> list[dict]:
    print(f"[Querying Dataverse API: {API_FILES_URL}")
    try:
        resp = requests.get(API_FILES_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        files = data.get("data", [])
        print(f"Found {len(files)} files via API.")
        return files
    except Exception as e:
        print(f"API query failed ({e}). Falling back to known file list.")
        return []


def build_download_url(file_id: int) -> str:
    return f"{BASE_URL}/api/access/datafile/{file_id}"


def download_file(url: str, dest: Path, resume: bool = False) -> bool:
    headers = {}
    initial_size = 0

    if resume and dest.exists():
        initial_size = dest.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=60)

        # 416 = Range Not Satisfiable → file already complete
        if resp.status_code == 416:
            print(f"  SKIP {dest.name} already complete.")
            return True

        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) + initial_size

        mode = "ab" if resume and initial_size > 0 else "wb"
        with open(dest, mode) as f, tqdm(
            desc=f"  {dest.name}",
            total=total,
            initial=initial_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            ncols=80,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                bar.update(len(chunk))
        return True

    except requests.RequestException as e:
        print(f" Failed to download {dest.name}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    target = extract_to / zip_path.stem
    if target.exists():
        print(f" Already extracted → {target}")
        return
    print(f" {zip_path.name} → {target}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)
