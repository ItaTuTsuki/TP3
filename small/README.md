# ⚽ Premier League — Prédire le poste d'un joueur (ML classique)

TP3 Big Data — **dossier `small/` (partie 1)** : workflow complet de Machine Learning sur un petit dataset (pas de big data).

> **Question :** peut-on deviner si un joueur est défenseur, milieu ou attaquant uniquement à partir de ses statistiques ?
> **Réponse :** oui, dans environ **85 % des cas** (F1 macro de 0,85 sur le jeu de test, contre 33 % au hasard).

- Exemple Kaggle de départ : [English Premier League Players Statistics](https://www.kaggle.com/code/desalegngeb/english-premier-league-players-statistics/notebook) (EDA seule, prolongée ici sur tout le workflow)
- Dataset : [Premier League player statistics](https://www.kaggle.com/datasets/rishikeshkanabar/premier-league-player-statistics-updated-daily) (snapshot du 24/09/2020, 571 joueurs, 59 colonnes), copié dans `data/`

## Workflow

| Étape | Contenu | Où |
|---|---|---|
| **Data** | chargement du CSV | notebook §1 |
| **EDA** | postes, âges, nombre de matchs, stats clés par poste, corrélations | notebook §2 |
| **Prétraitement** | conversion des `%`, valeurs manquantes, exclusion des gardiens, filtre ≥ 10 matchs. **Piège détecté :** certaines stats ne sont renseignées que pour certains postes, et leur absence trahit le poste. Seules les 17 stats communes à tous les postes sont gardées. | notebook §3, `src/features.py` |
| **Transformation** | stats cumulées → **par match**, standardisation (`StandardScaler` dans un `Pipeline`) | notebook §4, `src/features.py` |
| **Machine Learning** | régression logistique, Random Forest, Gradient Boosting, recherche d'hyperparamètres par `GridSearchCV` ; **bonus :** clustering KMeans + ACP | notebook §5 |
| **Évaluation** | validation croisée stratifiée à 5 plis, F1 macro, matrice de confusion, importance par permutation, analyse des erreurs | notebook §6 |
| **Déploiement** | modèle sauvegardé (`models/`), **app Streamlit** et image **Docker** | `app/streamlit_app.py`, `Dockerfile` |

## Résultats

| Modèle | F1 macro (validation croisée) |
|---|---|
| Régression logistique | 0,866 |
| **Random Forest** (retenu) | **0,872** après optimisation |
| Gradient Boosting | 0,845 |

Sur le jeu de test : **accuracy 85,1 %**, F1 macro 0,852. Le poste le plus difficile est **milieu**, un poste hybride : des milieux défensifs comme Fernandinho ou Matić sont prédits défenseurs, des ailiers comme Mahrez ou Zaha sont prédits milieux.

## Structure

```
small/
├── data/epl_players_2020.csv      # dataset (96 Ko)
├── notebooks/01_epl_workflow.ipynb # tout le workflow, déjà exécuté
├── src/features.py                 # prétraitement + transformation (partagé notebook / app)
├── models/                         # modèle entraîné + métadonnées (métriques, features)
├── app/streamlit_app.py            # app de déploiement
├── Dockerfile                      # image de déploiement de l'app
├── requirements.txt                # dépendances de l'app
└── requirements-dev.txt            # + dépendances du notebook
```

## L'app Streamlit en bref

[Streamlit](https://streamlit.io) transforme un script Python en site web interactif (listes déroulantes, champs, graphiques), sans HTML ni JavaScript. L'app permet de :
- **Prédire** : choisir un joueur ou saisir des stats, puis voir le poste prédit et les probabilités ;
- **Explorer** : filtrer par club et repérer les joueurs dont le profil diffère de leur poste officiel.

## Lancer en local

```bash
cd small
python -m venv .venv
# Windows : .venv\Scripts\activate     Linux/Mac : source .venv/bin/activate
pip install -r requirements-dev.txt

jupyter notebook notebooks/01_epl_workflow.ipynb   # réexécuter le workflow (régénère models/)
streamlit run app/streamlit_app.py                  # lancer l'app
```

## Lancer avec Docker (sans installer Python)

```bash
cd small
docker build -t tp3-epl .
docker run -p 8501:8501 tp3-epl
```

Ouvrir ensuite http://localhost:8501. L'image contient Python, les dépendances, le modèle et l'app.

## Déployer l'app en ligne (gratuit)

1. Pousser le repo sur GitHub.
2. Aller sur [share.streamlit.io](https://share.streamlit.io), se connecter avec GitHub, puis cliquer sur **Create app**.
3. Choisir le repo `TP3`, la branche `main` et le fichier `small/app/streamlit_app.py`, puis cliquer sur **Deploy**.

Streamlit Cloud installe `requirements.txt` automatiquement. Le modèle est versionné dans `models/`, il n'y a donc rien à réentraîner.

## Pourquoi pas de big data ici ?

Environ 370 joueurs × 17 colonnes, soit quelques Ko : pandas et scikit-learn suffisent largement, et Spark n'apporterait que de la complexité. C'est l'inverse du dossier [`big/`](../big) (Transfermarkt, millions de lignes, Spark).
