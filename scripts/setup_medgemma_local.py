"""Download a checksummed Windows llama.cpp runtime and pinned Q4 MedGemma."""
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from multicare_data.common import Project, file_hash, safe_path, write_json

project = Project(Path(__file__).resolve().parents[1])
load_dotenv(project.path(".env"))
cfg = project.config("medgemma")
request = urllib.request.Request("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10",
                                 headers={"User-Agent": "multicare-dataset-foundation"})
with urllib.request.urlopen(request, timeout=60) as response:
    releases = json.load(response)
asset = next(asset for release in releases for asset in release["assets"]
             if asset["name"].endswith("bin-win-cpu-x64.zip"))
folder = project.path("models/llama_cpp")
folder.mkdir(parents=True, exist_ok=True)
archive = folder / asset["name"]
if not archive.exists():
    urllib.request.urlretrieve(asset["browser_download_url"], archive)
actual = file_hash(archive)
if asset.get("digest") and asset["digest"] != "sha256:" + actual:
    raise ValueError("llama.cpp archive checksum mismatch")
with zipfile.ZipFile(archive) as zipped:
    for item in zipped.infolist():
        safe_path(folder, item.filename)
    zipped.extractall(folder)
info = HfApi().model_info(cfg["gguf_repo"])
filename = cfg["gguf_filename"]
if filename not in {item.rfilename for item in info.siblings}:
    raise ValueError("Configured quantized MedGemma filename is missing")
path = hf_hub_download(cfg["gguf_repo"], filename, revision=info.sha,
                       local_dir=project.path("models/medgemma_gguf"))
lock = {"gguf_repo": cfg["gguf_repo"], "revision": info.sha, "filename": filename,
        "model_path": str(Path(path).relative_to(project.root)).replace("\\", "/"),
        "model_sha256": file_hash(Path(path)), "runtime_asset": asset["name"],
        "runtime_sha256": actual}
write_json(project.path("models/medgemma_local.lock.json"), lock)
print(json.dumps(lock, indent=2))
