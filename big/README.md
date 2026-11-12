# 💶 Transfermarkt — Estimer la valeur marchande des joueurs avec Spark

TP3 Big Data — **dossier `big/` (partie 2)** : workflow complet de Machine Learning **big data** avec Apache Spark (PySpark).

> **Question :** à partir des performances d'un joueur sur une saison et de son profil, peut-on estimer sa valeur marchande ?

- Dataset Kaggle : [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) — environ 930 Mo de CSV, **1,9 M de matchs joués**, 1,3 M d'événements de match, 656 000 valeurs marchandes, 50 000 joueurs (données arrêtées en juillet 2026).

## Workflow

| Étape | Contenu | Où |
|---|---|---|
| **Data** | session Spark, lecture des CSV, **conversion en Parquet** (taille divisée par environ 10) | notebook §1 |
| **EDA** | agrégations Spark : distribution (log), inflation des valeurs, poste, âge, championnat | notebook §2 |
| **Prétraitement** | valeurs manquantes, aberrantes, définition des saisons (juillet → juin) | notebook §3, `src/features.py` |
| **Transformation** | **1,9 M de matchs agrégés en stats joueur × saison** (`groupBy`, jointure broadcast), cible issue de la **dernière valeur de la saison** (fonctions de fenêtre), valeur précédente (`lag`) | notebook §4, `src/features.py` |
| **Machine Learning** | Pipelines **Spark ML** : régression linéaire, Random Forest, GBT ; scénario A (performances + profil) vs scénario B (+ valeur précédente) ; `CrossValidator` | notebook §5 |
| **Évaluation** | découpage **temporel** (train 2013-2022 / test 2023-2025), RMSE, MAE, R², erreur en €, importance des variables, erreurs par poste | notebook §6 |
| **Déploiement** | pipeline Spark sauvegardé, **job batch** `src/score.py` (`spark-submit`), **app Streamlit**, images **Docker** et `docker-compose` | `models/`, `src/score.py`, `app/`, `Dockerfile*` |

## Résultats

Dataset final : **79 642 lignes joueur × saison**, construites à partir de 1,9 M de matchs et 656 000 évaluations. Entraînement sur les saisons 2013-2022 (59 771 lignes), test sur 2023-2025 (19 871 lignes). La cible est `log(valeur)`.

| Scénario | Modèle | RMSE (log) | R² |
|---|---|---|---|
| A — performances + profil | Régression linéaire | 0,889 | 0,682 |
| | Random Forest | 0,841 | 0,716 |
| | GBT | 0,814 | 0,734 |
| B — A + valeur de la saison précédente | Régression linéaire | 0,589 | 0,861 |
| | Random Forest | 0,549 | 0,879 |
| | GBT | 0,487 | 0,905 |
| **B — GBT optimisé** (`CrossValidator`, maxDepth = 5, maxIter = 120) | | **0,482** | **0,906** |

Pour le modèle retenu, sur les saisons de test : **erreur relative médiane de 25 %**, erreur absolue médiane d'environ 375 k€, et 57 % des joueurs estimés à ±30 % de leur vraie valeur.
- Avec seulement les performances et le profil (scénario A), le modèle explique déjà 73 % de la variance.
- La valeur précédente apporte beaucoup, car Transfermarkt fait évoluer ses valeurs par petites touches.
- Les variables les plus importantes (GBT) sont la valeur précédente (51 %), l'âge (14 %), le championnat (13 %), puis le nombre de matchs et les minutes jouées.

## Architecture de déploiement

```
data/raw/*.csv ──► Parquet ──► src/features.py ──► Pipeline Spark ML ──► models/market_value_model/
                                     │                                          │
                                     └──────────► src/score.py (spark-submit) ◄─┘
                                                        │
                                                        ▼
                                   app/data/predictions.csv.gz ──► app/streamlit_app.py
```

L'entraînement et le scoring sont faits **en batch par Spark**. L'app web lit seulement les prédictions : elle n'a besoin ni de Java ni de Spark, et se déploie gratuitement.

## Structure

```
big/
├── scripts/download_data.py              # télécharge le dataset Kaggle dans data/raw/
├── notebooks/02_transfermarkt_spark.ipynb  # tout le workflow, déjà exécuté
├── src/features.py                       # préparation Spark (partagé notebook / scoring)
├── src/score.py                          # job de scoring batch (déploiement)
├── models/                               # pipeline Spark entraîné + métriques
├── app/streamlit_app.py                  # app de consultation
├── app/data/                             # prédictions produites par src/score.py
├── Dockerfile / Dockerfile.spark         # images app (légère) / Spark (Java + PySpark)
├── docker-compose.yml                    # score, app, notebook
├── requirements.txt                      # dépendances de l'app (sans Spark)
└── requirements-spark.txt                # + PySpark, pour le notebook et le scoring
```

## Lancer en local

**Prérequis :** Python 3.10+ et **Java 17 ou 21** (exigé par PySpark 4).
- Linux / Mac / WSL : `sudo apt install openjdk-21-jdk`
- Sous Windows natif, Spark demande en plus `winutils.exe`. **Le plus simple est d'utiliser WSL (Ubuntu).**

```bash
cd big
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-spark.txt

python scripts/download_data.py                        # ≈ 250 Mo à télécharger
jupyter notebook notebooks/02_transfermarkt_spark.ipynb  # workflow complet (≈ 15-20 min)
spark-submit src/score.py                              # ou : python src/score.py
streamlit run app/streamlit_app.py
```

### Avec Docker (sans installer Java ni Spark)

```bash
cd big
docker compose run --rm score   # télécharge les données si besoin, puis job Spark -> app/data/
docker compose up app           # app Streamlit -> http://localhost:8501
docker compose up notebook      # Jupyter + Spark -> http://localhost:8888
```

- `Dockerfile.spark` : Python, Java 17 et PySpark (notebook et job de scoring).
- `Dockerfile` : app Streamlit seule, image légère sans Spark.

### Sur Kaggle (sans rien installer)

1. Créer un notebook Kaggle, puis **Add Input** et choisir le dataset `davidcariboo/player-scores`.
2. Importer `notebooks/02_transfermarkt_spark.ipynb` (File → Import Notebook).
3. La première cellule télécharge automatiquement `big/src/features.py` depuis ce repo GitHub.
4. Activer Internet dans les paramètres, puis exécuter `!pip install pyspark` si PySpark est absent.

## Déployer l'app en ligne (gratuit)

1. Pousser le repo sur GitHub (les prédictions `app/data/` sont versionnées).
2. Aller sur [share.streamlit.io](https://share.streamlit.io) et cliquer sur **Create app**.
3. Choisir le repo `TP3` et le fichier `big/app/streamlit_app.py`, puis cliquer sur **Deploy**.

## Pourquoi Spark ?

- **Volume :** environ 930 Mo de CSV et des millions de lignes à joindre et agréger. Avec pandas, c'est lent et gourmand en mémoire ; Spark parallélise sur tous les cœurs.
- **Passage à l'échelle :** le même code tourne sur un cluster (YARN, Kubernetes, Databricks) en changeant seulement `master`. Il resterait donc valable pour des données 100× plus grosses, par exemple des données de tracking.
- **Outils big data :** format Parquet, jointures broadcast, fonctions de fenêtre, pipelines ML et validation croisée distribués.

À comparer avec le dossier [`small/`](../small) (Premier League, environ 400 lignes), où pandas et scikit-learn suffisent.
