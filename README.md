# TP3_ID06 — Deux cas personnels : small data et big data ⚽

**Auteur :** Hugo EXERTIER (ID 06), cours Big Data 5A (Mohamed Quafafou), rendu du 18/11/2026.

Deux études complètes autour du football, qui suivent le même workflow :

**Data → EDA → Prétraitement → Transformation → Machine Learning → Évaluation → Déploiement**

| | [`Small/`](Small) — small data | [`Big/`](Big) — big data |
|---|---|---|
| **Question** | Deviner le **poste** d'un joueur de Premier League à partir de ses stats | Estimer la **valeur marchande** d'un joueur à partir de sa saison |
| **Notebook d'origine** | [EPL Players Statistics](https://www.kaggle.com/code/desalegngeb/english-premier-league-players-statistics/notebook) (Kaggle, EDA pandas) | [football-player-market-value-prediction](https://github.com/Anubhav1989/football-player-market-value-prediction) (8 notebooks pandas / scikit-learn) |
| **Dataset Kaggle** | [Premier League player statistics](https://www.kaggle.com/datasets/rishikeshkanabar/premier-league-player-statistics-updated-daily) | [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) |
| **Volume** | 571 joueurs, 96 Ko | ≈ 930 Mo, 1,9 M de matchs, 656 000 évaluations |
| **Outils** | pandas, scikit-learn, + complément **PySpark** (Spark SQL, Spark ML) | **Apache Spark (PySpark)** : Parquet, Spark SQL, Spark ML |
| **Type de ML** | classification (+ clustering) | régression |
| **Meilleur modèle** | Random Forest : F1 = 0,85, accuracy 85 % | GBT : R² = 0,91, erreur médiane 25 % |
| **Déploiement** | app Streamlit + Docker | job Spark batch + app Streamlit + Docker Compose |
| **Notebook** | [`Small/notebooks/01_epl_workflow.ipynb`](Small/notebooks/01_epl_workflow.ipynb) | [`Big/notebooks/02_transfermarkt_spark.ipynb`](Big/notebooks/02_transfermarkt_spark.ipynb) |

Chaque dossier a son propre README avec les résultats détaillés et les commandes pour lancer le projet.

## Consignes de rendu

- **Première cellule de chaque notebook** : titre du TP, puis la contribution : le notebook s'exécute-t-il de bout en bout, problèmes rencontrés dans le notebook d'origine, modifications apportées, autres informations.
- **Étiquetage du processus IA** : chaque section est précédée d'une cellule `## [Étape]` : `[Data]`, `[EDA]`, `[Prétraitement]`, `[Transformation]`, `[ML]`, `[Éval]`, `[Déploiement]`, plus `[Conclusion]`.
- **PySpark** : le notebook d'origine de la partie Small n'utilise pas PySpark. La section 8 du notebook Small refait donc la lecture, le prétraitement, la transformation, le ML et l'évaluation avec Spark, et compare avec scikit-learn. Les notebooks d'origine de la partie Big n'utilisent pas PySpark non plus : tout le notebook Big est fait avec Spark.

## Démarrage rapide

Prérequis : Python 3.11+ (et Java 17+ seulement pour relancer Spark).

```bash
# Small : notebook + app (Python seul suffit, Java 17+ pour la section PySpark)
cd Small
pip install -r requirements-dev.txt
streamlit run app/streamlit_app.py

# Big : app seule (les prédictions Spark sont déjà dans Big/app/data/)
cd ../Big
pip install -r requirements.txt
streamlit run app/streamlit_app.py

# Big : refaire tout le pipeline Spark (Java 17+ requis, ou Docker)
pip install -r requirements-spark.txt
python scripts/download_data.py
jupyter notebook notebooks/02_transfermarkt_spark.ipynb
```

## Pourquoi deux approches ?

- **Small** : quelques centaines de lignes tiennent en mémoire. pandas et scikit-learn sont plus simples et plus rapides, et Spark n'apporterait que de la complexité.
- **Big** : il faut joindre et agréger des millions de lignes. Spark parallélise le travail sur tous les cœurs, utilise des formats adaptés (Parquet) et le même code passerait sur un cluster pour des données bien plus grosses.

## Limites

- **Small** : stats de carrière figées au 24/09/2020. Le poste officiel est une étiquette grossière (les milieux sont un poste hybride).
- **Big** : la valeur Transfermarkt est une estimation d'experts, pas un prix de transfert. Le modèle sous-estime les toutes meilleures stars (effet de « retour à la moyenne »). Les données s'arrêtent en juillet 2026.
