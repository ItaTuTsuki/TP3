"""Télécharge le dataset Transfermarkt dans data/raw/ (≈ 240 Mo zippé, ≈ 870 Mo décompressé).

    python scripts/download_data.py

Le dataset est public. On essaie d'abord Kaggle (davidcariboo/player-scores) ; en 2026,
Kaggle refuse les téléchargements sans compte (erreur 404). On utilise alors le miroir
officiel de l'auteur du dataset (lien « Download Dataset » du dépôt
https://github.com/dcaribou/transfermarkt-datasets), qui contient les mêmes tables en .csv.gz.

Avec un compte Kaggle, on peut aussi utiliser la CLI :
    kaggle datasets download davidcariboo/player-scores -p data/raw --unzip
"""
import gzip
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DATASET = "davidcariboo/player-scores"
URLS = [
    f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}",
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.zip",
]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(archive: Path) -> None:
    for url in URLS:
        print(f"Téléchargement depuis {url} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(archive, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            if zipfile.is_zipfile(archive):
                return
            print("  réponse invalide (pas une archive zip)")
        except urllib.error.URLError as e:
            print(f"  échec : {e}")
    raise SystemExit("Téléchargement impossible : utiliser la CLI Kaggle (voir l'en-tête du script).")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if (RAW_DIR / "appearances.csv").exists():
        print(f"Déjà présent : {RAW_DIR}")
        return
    archive = RAW_DIR / "archive.zip"
    download(archive)
    print("Décompression ...")
    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW_DIR)
    archive.unlink()
    # Le miroir fournit des .csv.gz : on les décompresse en .csv
    for gz in RAW_DIR.glob("*.csv.gz"):
        with gzip.open(gz, "rb") as src, open(RAW_DIR / gz.stem, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        gz.unlink()
    for f in sorted(RAW_DIR.glob("*.csv")):
        print(f"  {f.name:28s} {f.stat().st_size / 1e6:8.1f} Mo")


if __name__ == "__main__":
    main()
