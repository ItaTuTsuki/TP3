"""App Streamlit — prédire le poste d'un joueur de Premier League.

Lancer :  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import features as F  # noqa: E402

POSITION_FR = {"Defender": "Défenseur", "Midfielder": "Milieu", "Forward": "Attaquant"}

st.set_page_config(page_title="Poste d'un joueur EPL", page_icon="⚽", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "models" / "position_model.joblib")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    return model, meta


@st.cache_data
def load_players():
    return F.prepare()


model, meta = load_model()
players = load_players()

st.title("⚽ Quel est le poste de ce joueur ?")
st.caption(
    f"Modèle : **{meta['model']}** entraîné sur {meta['n_train_rows']} joueurs de Premier League "
    f"(F1 test = {meta['test_f1_macro']:.2f}, accuracy test = {meta['test_accuracy']:.0%}). "
    "Le modèle ne voit que des statistiques par match, jamais le poste."
)

tab_predict, tab_explore = st.tabs(["🔮 Prédire", "🔎 Explorer les joueurs"])

with tab_predict:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("1. Point de départ")
        names = ["(saisie libre)"] + sorted(players["Name"])
        choice = st.selectbox("Pré-remplir avec un joueur existant", names, index=names.index("Kevin De Bruyne") if "Kevin De Bruyne" in names else 0)
        base = players[players["Name"] == choice].iloc[0] if choice != "(saisie libre)" else None

        st.subheader("2. Statistiques cumulées")
        appearances = st.number_input("Matchs joués", min_value=1, value=int(base["Appearances"]) if base is not None else 50, key=f"{choice}-apps")
        totals = {}
        cols = st.columns(3)
        for i, stat in enumerate(F.COUNT_STATS):
            default = int(base[stat]) if base is not None else 0
            totals[stat] = cols[i % 3].number_input(stat, min_value=0, value=default, key=f"{choice}-{stat}")

    with right:
        X = F.per_match_from_totals(totals, appearances)
        proba = pd.Series(model.predict_proba(X)[0], index=model.classes_)
        pred = proba.idxmax()
        st.subheader("3. Prédiction")
        st.metric("Poste prédit", POSITION_FR[pred], f"{proba[pred]:.0%} de confiance", delta_color="off")
        if base is not None:
            real = base["Position"]
            ok = "✅" if real == pred else "❌"
            st.write(f"Poste officiel : **{POSITION_FR[real]}** {ok}")
        st.bar_chart(proba.rename(index=POSITION_FR).sort_values(), horizontal=True)
        with st.expander("Stats par match envoyées au modèle"):
            st.dataframe(X.T.rename(columns={0: "valeur"}).style.format("{:.3f}"))

with tab_explore:
    X_all = players[F.FEATURES]
    explore = players[["Name", "Club", "Position", "Appearances"]].copy()
    explore["Poste prédit"] = model.predict(X_all)
    explore["Confiance"] = model.predict_proba(X_all).max(axis=1)
    explore["Position"] = explore["Position"].map(POSITION_FR)
    explore["Poste prédit"] = explore["Poste prédit"].map(POSITION_FR)

    c1, c2 = st.columns(2)
    club = c1.selectbox("Club", ["Tous"] + sorted(explore["Club"].unique()))
    only_diff = c2.checkbox("Seulement les joueurs dont le profil diffère du poste officiel")
    view = explore if club == "Tous" else explore[explore["Club"] == club]
    if only_diff:
        view = view[view["Position"] != view["Poste prédit"]]
    st.caption("Prédictions faites par le modèle final (entraîné sur tous les joueurs) : "
               "voir le notebook pour l'évaluation honnête sur données de test.")
    st.dataframe(view.sort_values("Appearances", ascending=False),
                 column_config={"Confiance": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")},
                 hide_index=True, width="stretch")
