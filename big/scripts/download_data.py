"""Télécharge le dataset Kaggle Transfermarkt dans data/raw/ (≈ 250 Mo zippé, ≈ 930 Mo décompressé).

    python scripts/download_data.py

Le dataset est public : pas besoin de compte Kaggle. Si le téléchargement direct
échoue, utiliser la CLI Kaggle : kaggle datasets download davidcariboo/player-scores -p data/raw --unzip
"""
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASET = "davidcariboo/player-scores"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if (RAW_DIR / "appearances.csv").exists():
        print(f"Déjà présent : {RAW_DIR}")
        return
    archive = RAW_DIR / "archive.zip"
    print(f"Téléchargement de {DATASET} ...")
    with urllib.request.urlopen(URL) as resp, open(archive, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    print("Décompression ...")
    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW_DIR)
    archive.unlink()
    for f in sorted(RAW_DIR.glob("*.csv")):
        print(f"  {f.name:28s} {f.stat().st_size / 1e6:8.1f} Mo")


if __name__ == "__main__":
    main()
