"""App Streamlit — valeur marchande estimée par le modèle Spark.

Lancer :  streamlit run app/streamlit_app.py

L'app lit les prédictions produites par le job Spark `src/score.py`
(app/data/*.csv.gz) : elle n'a pas besoin de Spark ni de Java, donc se déploie
facilement (Streamlit Community Cloud).
"""
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
POSITION_FR = {"Attack": "Attaquant", "Midfield": "Milieu", "Defender": "Défenseur", "Goalkeeper": "Gardien"}

st.set_page_config(page_title="Valeur marchande — Spark", page_icon="💶", layout="wide")


@st.cache_data
def load():
    pred = pd.read_csv(DATA / "predictions.csv.gz")
    pred["position"] = pred["position"].map(POSITION_FR).fillna(pred["position"])
    pred["league_name"] = pred["league_name"].fillna(pred["league"])
    hist = pd.read_csv(DATA / "history.csv.gz", parse_dates=["date"])
    return pred, hist


def eur(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.1f} M€"
    return f"{x / 1e3:.0f} k€"


pred, hist = load()
season = int(pred["season"].max())

st.title("💶 Valeur marchande des joueurs — estimation par Spark ML")
st.caption(
    f"Saison {season}/{(season + 1) % 100:02d} · {len(pred):,} joueurs · modèle GBT entraîné avec PySpark "
    "sur ~80 000 saisons-joueurs (2013-2022) construites à partir de 1,9 M de matchs Transfermarkt.".replace(",", " ")
)

with st.sidebar:
    st.header("Filtres")
    leagues = st.multiselect("Championnat", sorted(pred["league_name"].unique()))
    positions = st.multiselect("Poste", sorted(pred["position"].unique()))
    age = st.slider("Âge", 15, 45, (15, 45))
    min_minutes = st.slider("Minutes jouées minimum", 0, 4000, 900, step=100)
    min_value = st.select_slider("Valeur Transfermarkt minimum",
                                 options=[0, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
                                 value=1_000_000, format_func=lambda v: eur(v) if v else "0 €")

view = pred[pred["age"].between(*age) & (pred["minutes"] >= min_minutes) & (pred["market_value"] >= min_value)]
if leagues:
    view = view[view["league_name"].isin(leagues)]
if positions:
    view = view[view["position"].isin(positions)]

tab_player, tab_deals, tab_table = st.tabs(["🔎 Joueur", "📈 Sous / sur-évalués", "📋 Tableau"])

with tab_player:
    names = view.sort_values("market_value", ascending=False)["name"] + " — " + view.sort_values("market_value", ascending=False)["club_name"]
    if names.empty:
        st.info("Aucun joueur ne correspond aux filtres.")
    else:
        pick = st.selectbox("Joueur", names.tolist())
        p = view.loc[names[names == pick].index[0]]
        c0, c1, c2, c3 = st.columns([1, 2, 2, 2])
        if isinstance(p["image_url"], str):
            c0.image(p["image_url"], width=110)
        c1.metric("Valeur Transfermarkt", eur(p["market_value"]))
        c2.metric("Valeur estimée (modèle)", eur(p["predicted_value"]), f"{p['ratio'] - 1:+.0%}")
        c3.metric("Saison", f"{int(p['matches'])} matchs · {int(p['goals'])} buts · {int(p['assists'])} passes déc.")
        st.write(f"**{p['position']}** ({p['sub_position']}) · {p['age']:.0f} ans · {p['club_name']} · {p['league_name']}")
        h = hist[hist["player_id"] == p["player_id"]].set_index("date")["market_value_in_eur"] / 1e6
        st.line_chart(h.rename("Valeur Transfermarkt (M€)"))

with tab_deals:
    st.write("Écart entre l'estimation du modèle et la valeur Transfermarkt : un ratio > 1 signifie que "
             "les performances de la saison justifieraient une valeur **plus élevée** que celle affichée.")
    cols = ["name", "club_name", "league_name", "position", "age", "minutes", "goals", "assists",
            "market_value", "predicted_value", "ratio"]
    fmt = {"market_value": st.column_config.NumberColumn("Valeur TM (€)", format="%.0f"),
           "predicted_value": st.column_config.NumberColumn("Valeur estimée (€)", format="%.0f"),
           "ratio": st.column_config.NumberColumn("Ratio", format="%.2f")}
    left, right = st.columns(2)
    left.subheader("Potentiellement sous-évalués")
    left.dataframe(view.nlargest(20, "ratio")[cols], column_config=fmt, hide_index=True)
    right.subheader("Potentiellement sur-évalués")
    right.dataframe(view.nsmallest(20, "ratio")[cols], column_config=fmt, hide_index=True)
    st.subheader("Valeur réelle vs estimée")
    st.scatter_chart(view.assign(**{"Valeur TM (M€)": view["market_value"] / 1e6,
                                    "Valeur estimée (M€)": view["predicted_value"] / 1e6}),
                     x="Valeur TM (M€)", y="Valeur estimée (M€)", color="position")

with tab_table:
    st.dataframe(view.drop(columns=["image_url", "player_id"]).sort_values("market_value", ascending=False),
                 hide_index=True, width="stretch")
