"""Déploiement batch : applique le modèle Spark sauvegardé à la dernière saison.

    spark-submit src/score.py            # ou : python src/score.py
    python src/score.py --season 2024    # autre saison

Sorties (lues par l'app Streamlit, qui n'a donc pas besoin de Spark) :
  app/data/predictions.csv.gz  une ligne par joueur : valeur réelle vs valeur estimée
  app/data/history.csv.gz      historique des valeurs marchandes de ces joueurs
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyspark.ml import PipelineModel  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from src import features as T  # noqa: E402

MODEL_DIR = ROOT / "models" / "market_value_model"
# Le modèle est versionné en zip : Spark génère des noms de fichiers très longs qui
# dépassent la limite de 260 caractères des chemins Windows lors d'un git clone.
MODEL_ZIP = ROOT / "models" / "market_value_model.zip"
OUT_DIR = ROOT / "app" / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, help="saison à scorer (défaut : la plus récente)")
    args = parser.parse_args()

    spark = T.get_spark("transfermarkt-scoring")
    tables = T.load_tables(spark)
    data = T.build_dataset(tables)
    season = args.season or data.agg(F.max("season")).first()[0]
    data = data.filter(F.col("season") == season)

    if not MODEL_DIR.exists():
        shutil.unpack_archive(MODEL_ZIP, MODEL_DIR)
    model = PipelineModel.load(str(MODEL_DIR))
    leagues = tables["competitions"].select(
        F.col("competition_id").alias("league"),
        F.concat_ws(" — ", F.initcap(F.regexp_replace("name", "-", " ")), "country_name").alias("league_name"),
    )
    pred = (
        model.transform(data)
        .withColumn("predicted_value", F.expm1("prediction"))
        .join(F.broadcast(leagues), "league", "left")
        .select(
            "player_id", "name", "club_name", "league", "league_name", "position", "sub_position", "foot",
            F.round("age", 1).alias("age"), "season", "matches", "minutes", "goals", "assists", "matches_europe",
            "market_value", F.round("predicted_value", -3).alias("predicted_value"), "image_url",
        )
        .withColumn("ratio", F.round(F.col("predicted_value") / F.col("market_value"), 3))
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = pred.toPandas()
    pdf.to_csv(OUT_DIR / "predictions.csv.gz", index=False)

    history = (
        tables["player_valuations"]
        .join(F.broadcast(pred.select("player_id")), "player_id")
        .select("player_id", "date", "market_value_in_eur")
        .orderBy("player_id", "date")
        .toPandas()
    )
    history.to_csv(OUT_DIR / "history.csv.gz", index=False)
    print(f"Saison {season} : {len(pdf)} joueurs scorés -> {OUT_DIR}")
    spark.stop()


if __name__ == "__main__":
    main()
