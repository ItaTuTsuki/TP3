# TP3 — Machine Learning : petites données vs big data ⚽

Deux études complètes autour du football, qui suivent le même workflow :

**Data → EDA → Prétraitement → Transformation → Machine Learning → Évaluation → Déploiement**

| | [`small/`](small) — Data | [`big/`](big) — Big Data |
|---|---|---|
| **Question** | Deviner le **poste** d'un joueur de Premier League à partir de ses stats | Estimer la **valeur marchande** d'un joueur à partir de sa saison |
| **Source Kaggle** | [EPL Players Statistics](https://www.kaggle.com/code/desalegngeb/english-premier-league-players-statistics/notebook) | [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) |
| **Volume** | 571 joueurs, 96 Ko | ≈ 930 Mo, 1,9 M de matchs, 656 000 évaluations |
| **Outils** | pandas, scikit-learn | **Apache Spark (PySpark)** : Parquet, Spark SQL, Spark ML |
| **Type de ML** | classification (+ clustering) | régression |
| **Meilleur modèle** | Random Forest : F1 = 0,85, accuracy 85 % | GBT : R² = 0,91, erreur médiane 25 % |
| **Déploiement** | app Streamlit + Docker | job Spark batch + app Streamlit + Docker Compose |
| **Notebook** | [`small/notebooks/01_epl_workflow.ipynb`](small/notebooks/01_epl_workflow.ipynb) | [`big/notebooks/02_transfermarkt_spark.ipynb`](big/notebooks/02_transfermarkt_spark.ipynb) |

Chaque dossier a son propre README avec les résultats détaillés et les commandes pour lancer le projet.

## Démarrage rapide

Prérequis : Python 3.11+ (et Java 17+ seulement pour relancer Spark).

```bash
# small : notebook + app (Python seul suffit)
cd small
pip install -r requirements-dev.txt
streamlit run app/streamlit_app.py

# big : app seule (les prédictions Spark sont déjà dans big/app/data/)
cd ../big
pip install -r requirements.txt
streamlit run app/streamlit_app.py

# big : refaire tout le pipeline Spark (Java 17+ requis, ou Docker)
pip install -r requirements-spark.txt
python scripts/download_data.py
jupyter notebook notebooks/02_transfermarkt_spark.ipynb
```

## Pourquoi deux approches ?

- **small** : quelques centaines de lignes tiennent en mémoire. pandas et scikit-learn sont plus simples et plus rapides, et Spark n'apporterait que de la complexité.
- **big** : il faut joindre et agréger des millions de lignes. Spark parallélise le travail sur tous les cœurs, utilise des formats adaptés (Parquet) et le même code passerait sur un cluster pour des données bien plus grosses.

## Limites

- **small** : stats de carrière figées au 24/09/2020. Le poste officiel est une étiquette grossière (les milieux sont un poste hybride).
- **big** : la valeur Transfermarkt est une estimation d'experts, pas un prix de transfert. Le modèle sous-estime les toutes meilleures stars (effet de « retour à la moyenne »). Les données s'arrêtent en juillet 2026.
