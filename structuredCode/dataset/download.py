import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


CHUNK_SIZE = 1024 * 1024

def get_file_list_from_api(api_files_url) -> list[dict]:
    print(f"[Querying Dataverse API: {api_files_url}")
    try:
        resp = requests.get(api_files_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        files = data.get("data", [])
        print(f"Found {len(files)} files via API.")
        return files
    except Exception as e:
        print(f"API query failed ({e}). Falling back to known file list.")
        return []

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


