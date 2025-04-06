import pandas as pd
import os
import mlflow
from sklearn.model_selection import train_test_split

# Caminhos dos dados
raw_dir = "./data/raw"
processed_dir = "./data/processed"
os.makedirs(processed_dir, exist_ok=True)


mlflow.set_experiment("PreparacaoDados")

with mlflow.start_run(run_name="PreparacaoDados"):
    dev_path = os.path.join(raw_dir, "dataset_kobe_dev.parquet")
    df_dev = pd.read_parquet(dev_path)

    colunas = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    df_dev = df_dev[colunas]

    df_dev = df_dev.dropna()

    filtered_path = os.path.join(processed_dir, "data_filtered.parquet")
    df_dev.to_parquet(filtered_path, index=False)

    X = df_dev.drop("shot_made_flag", axis=1)
    y = df_dev["shot_made_flag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_df = X_train.copy()
    train_df["shot_made_flag"] = y_train
    test_df = X_test.copy()
    test_df["shot_made_flag"] = y_test


    train_path = os.path.join(processed_dir, "base_train.parquet")
    test_path = os.path.join(processed_dir, "base_test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 123)
    mlflow.log_param("colunas_utilizadas", colunas)
    mlflow.log_metric("train_size", len(train_df))
    mlflow.log_metric("test_size", len(test_df))

