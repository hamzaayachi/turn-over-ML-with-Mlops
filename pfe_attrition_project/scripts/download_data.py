"""
Télécharge le jeu de données IBM HR Analytics (attrition) si absent.
Référencé par dvc.yaml (stage download_data).
"""
from pathlib import Path
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Miroir public (même schéma que le dataset IBM / Kaggle classique)
DATA_URL = (
    "https://raw.githubusercontent.com/dasarpai/DAI-Datasets/main/"
    + FILENAME
)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / FILENAME

    if dest.exists() and dest.stat().st_size > 0:
        print(f"OK: {dest} existe déjà ({dest.stat().st_size} octets).")
        return 0

    print(f"Téléchargement depuis {DATA_URL} …")
    urllib.request.urlretrieve(DATA_URL, dest)
    print(f"OK: fichier enregistré sous {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
