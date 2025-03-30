import pandas as pd
import os
import mlflow
from sklearn.model_selection import train_test_split

# Caminhos dos dados
raw_dir = "./data/raw"
processed_dir = "./data/processed"
os.makedirs(processed_dir, exist_ok=True)

# Nome da rodada no MLFlow
mlflow.set_experiment("PreparacaoDados")

with mlflow.start_run(run_name="PreparacaoDados"):
    # Carregando os dados
    dev_path = os.path.join(raw_dir, "dataset_kobe_dev.parquet")
    df_dev = pd.read_parquet(dev_path)

    # Selecionar colunas relevantes
    colunas = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    df_dev = df_dev[colunas]

    # Remover linhas com valores faltantes
    df_dev = df_dev.dropna()

    # Salvar dataset filtrado
    filtered_path = os.path.join(processed_dir, "data_filtered.parquet")
    df_dev.to_parquet(filtered_path, index=False)

    # Separar em treino e teste
    X = df_dev.drop("shot_made_flag", axis=1)
    y = df_dev["shot_made_flag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_df = X_train.copy()
    train_df["shot_made_flag"] = y_train
    test_df = X_test.copy()
    test_df["shot_made_flag"] = y_test

    # Salvar datasets
    train_path = os.path.join(processed_dir, "base_train.parquet")
    test_path = os.path.join(processed_dir, "base_test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # Log de parâmetros e métricas
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("train_size", len(train_df))
    mlflow.log_metric("test_size", len(test_df))
