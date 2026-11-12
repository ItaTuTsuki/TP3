"""Prétraitement et transformation des stats Premier League.

Partagé entre le notebook (entraînement) et l'app Streamlit (prédiction),
pour garantir que les deux appliquent exactement les mêmes transformations.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "epl_players_2020.csv"

# Postes à prédire (les gardiens sont exclus : leurs stats les identifient à 100 %)
POSITIONS = ["Defender", "Midfielder", "Forward"]

# Seules les stats renseignées pour TOUS les postes de champ sont gardées.
# Sur le site de la Premier League, certaines stats n'existent que pour certains
# postes (ex : "Shots" absent chez 98 % des défenseurs, "Recoveries" absent chez
# 96 % des attaquants) : les utiliser ferait fuiter le poste via les valeurs manquantes.
COUNT_STATS = [
    "Goals",
    "Headed goals",
    "Goals with right foot",
    "Goals with left foot",
    "Hit woodwork",
    "Assists",
    "Big chances created",
    "Passes",
    "Crosses",
    "Tackles",
    "Blocked shots",
    "Interceptions",
    "Clearances",
    "Headed Clearance",
    "Fouls",
    "Offsides",
    "Yellow cards",
]

# Les stats cumulées dépendent du nombre de matchs : on les ramène "par match"
FEATURES = [f"{c} / match" for c in COUNT_STATS]

MIN_APPEARANCES = 10


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Prétraitement : types, valeurs manquantes, filtrage."""
    df = df.copy()
    # Colonnes en pourcentage stockées en texte ("81%")
    for col in [c for c in df.columns if c.endswith("%")]:
        df[col] = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")
    df = df.dropna(subset=["Age", "Nationality"])
    df = df[df["Position"].isin(POSITIONS)]
    df = df[df["Appearances"] >= MIN_APPEARANCES]
    # Pour les stats retenues, un vide = 0 événement (moins de 1 % des cas)
    df[COUNT_STATS] = df[COUNT_STATS].fillna(0)
    return df.reset_index(drop=True)


def add_per_match(df: pd.DataFrame) -> pd.DataFrame:
    """Transformation : stats cumulées -> stats par match."""
    df = df.copy()
    for col in COUNT_STATS:
        df[f"{col} / match"] = df[col] / df["Appearances"]
    return df


def prepare(path=DATA_PATH) -> pd.DataFrame:
    return add_per_match(clean(load_raw(path)))


def per_match_from_totals(totals: dict, appearances: int) -> pd.DataFrame:
    """Construit une ligne de features à partir de totaux saisis (app Streamlit)."""
    appearances = max(int(appearances), 1)
    row = {f"{c} / match": float(totals.get(c, 0)) / appearances for c in COUNT_STATS}
    return pd.DataFrame([row], columns=FEATURES).astype(np.float64)
