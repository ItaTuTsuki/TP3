"""Préparation des données Transfermarkt avec PySpark.

Partagé entre le notebook (entraînement) et `src/score.py` (déploiement batch),
pour que les deux construisent exactement les mêmes features.

Une ligne du dataset final = un joueur x une saison :
  - features : stats agrégées de la saison (matchs, minutes, buts…), profil (âge, poste…), championnat
  - cible    : dernière valeur marchande Transfermarkt connue sur la saison (log)
"""
import glob
import os
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]

TABLES = ["appearances", "player_valuations", "players", "competitions", "clubs", "games", "transfers"]

FIRST_SEASON = 2013  # appearances commencent en juillet 2012 -> 2013 a une saison précédente complète

CATEGORICAL = ["position", "sub_position", "foot", "league"]
NUMERIC_BASE = [
    "season", "age", "height_in_cm",
    "matches", "minutes", "goals", "assists", "yellow_cards", "red_cards",
    "goals_per90", "assists_per90", "minutes_per_match",
    "matches_league", "matches_cup", "matches_europe", "starter_share",
]
NUMERIC_WITH_HISTORY = NUMERIC_BASE + ["prev_log_value"]
LABEL = "log_value"


# --------------------------------------------------------------------------- Spark / données
def get_spark(app_name="transfermarkt", memory="4g") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", memory)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def find_data_dir() -> Path:
    """Dossier des CSV : variable TM_DATA_DIR, sinon Kaggle (/kaggle/input), sinon data/raw."""
    if os.environ.get("TM_DATA_DIR"):
        return Path(os.environ["TM_DATA_DIR"])
    kaggle = glob.glob("/kaggle/input/**/appearances.csv", recursive=True)
    if kaggle:
        return Path(kaggle[0]).parent
    return ROOT / "data" / "raw"


def csv_to_parquet(spark: SparkSession, csv_dir: Path, parquet_dir: Path) -> None:
    """Convertit chaque CSV en Parquet (format colonne compressé, typé, bien plus rapide à relire)."""
    for name in TABLES:
        out = parquet_dir / name
        if (out / "_SUCCESS").exists():  # déjà converti (et écriture terminée)
            continue
        (spark.read.csv(str(csv_dir / f"{name}.csv"), header=True, inferSchema=True)
         .write.mode("overwrite").parquet(str(out)))


def load_tables(spark: SparkSession, csv_dir: Path | None = None, parquet_dir: Path | None = None) -> dict:
    """Charge les tables : Parquet si disponible, sinon CSV."""
    csv_dir = csv_dir or find_data_dir()
    parquet_dir = parquet_dir or ROOT / "data" / "parquet"
    tables = {}
    for name in TABLES:
        if (parquet_dir / name / "_SUCCESS").exists():
            tables[name] = spark.read.parquet(str(parquet_dir / name))
        else:
            tables[name] = spark.read.csv(str(csv_dir / f"{name}.csv"), header=True, inferSchema=True)
    return tables


# --------------------------------------------------------------------------- transformations
def season_col(date_col: str = "date"):
    """Saison européenne : de juillet Y à juin Y+1 -> saison Y."""
    d = F.col(date_col)
    return F.when(F.month(d) >= 7, F.year(d)).otherwise(F.year(d) - 1)


def season_stats(appearances: DataFrame, competitions: DataFrame, games: DataFrame | None = None) -> DataFrame:
    """Agrège ~1,9 M de lignes de matchs en stats par (joueur, saison)."""
    comp = competitions.select(
        "competition_id",
        F.col("type").alias("comp_type"),
        F.col("sub_type").alias("comp_sub_type"),
    )
    apps = (
        appearances.join(F.broadcast(comp), "competition_id", "left")
        .withColumn("season", season_col("date"))
        .withColumn("is_league", F.coalesce(F.col("comp_type") == "domestic_league", F.lit(False)).cast("int"))
        .withColumn("is_cup", F.coalesce(F.col("comp_type") == "domestic_cup", F.lit(False)).cast("int"))
        .withColumn("is_europe", F.coalesce(F.col("comp_sub_type").startswith("uefa_"), F.lit(False)).cast("int"))
        .withColumn("is_starter", (F.col("minutes_played") >= 60).cast("int"))
    )
    agg = apps.groupBy("player_id", "season").agg(
        F.count("*").alias("matches"),
        F.sum("minutes_played").alias("minutes"),
        F.sum("goals").alias("goals"),
        F.sum("assists").alias("assists"),
        F.sum("yellow_cards").alias("yellow_cards"),
        F.sum("red_cards").alias("red_cards"),
        F.sum("is_league").alias("matches_league"),
        F.sum("is_cup").alias("matches_cup"),
        F.sum("is_europe").alias("matches_europe"),
        F.avg("is_starter").alias("starter_share"),
    )
    per90 = F.greatest(F.col("minutes"), F.lit(1)) / 90
    return (
        agg.withColumn("goals_per90", F.col("goals") / per90)
        .withColumn("assists_per90", F.col("assists") / per90)
        .withColumn("minutes_per_match", F.col("minutes") / F.col("matches"))
    )


def season_values(valuations: DataFrame) -> DataFrame:
    """Cible : dernière valeur marchande de chaque joueur sur chaque saison (+ celle de la saison précédente)."""
    v = (
        valuations.filter(F.col("market_value_in_eur") > 0)
        .withColumn("season", season_col("date"))
    )
    last = Window.partitionBy("player_id", "season").orderBy(F.col("date").desc())
    v = (
        v.withColumn("rn", F.row_number().over(last)).filter("rn = 1")
        .select(
            "player_id", "season",
            F.col("date").alias("value_date"),
            F.col("market_value_in_eur").alias("market_value"),
            F.coalesce(F.col("player_club_domestic_competition_id"), F.lit("Autre")).alias("league"),
            F.col("current_club_name").alias("club_name"),
        )
        .withColumn(LABEL, F.log1p("market_value"))
    )
    prev = Window.partitionBy("player_id").orderBy("season")
    return v.withColumn(
        "prev_log_value",
        F.when(F.lag("season").over(prev) == F.col("season") - 1, F.lag(LABEL).over(prev)),
    )


def player_profile(players: DataFrame) -> DataFrame:
    return players.select(
        "player_id", "name", "image_url", "date_of_birth", "height_in_cm",
        F.when(F.col("position") == "Missing", None).otherwise(F.col("position")).alias("position"),
        "sub_position", "foot",
    )


def build_dataset(tables: dict, min_minutes: int = 90) -> DataFrame:
    """Dataset final joueur x saison prêt pour le ML."""
    stats = season_stats(tables["appearances"], tables["competitions"])
    values = season_values(tables["player_valuations"])
    profile = player_profile(tables["players"])
    df = (
        values.join(stats, ["player_id", "season"], "inner")
        .join(profile, "player_id", "left")
        .withColumn("age", F.datediff("value_date", F.to_date("date_of_birth")) / 365.25)
        .filter(F.col("season") >= FIRST_SEASON)
        .filter(F.col("minutes") >= min_minutes)
        .filter(F.col("age").between(15, 45))
        .filter(F.col("position").isNotNull())
    )
    return df.drop("date_of_birth")
