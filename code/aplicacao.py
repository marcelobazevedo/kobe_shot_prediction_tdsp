import pandas as pd
import os
import mlflow
from pycaret.classification import load_model, predict_model
from sklearn.metrics import f1_score
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aplicar_modelo():
    processed_dir = "./data/processed"
    model_path = os.path.join(processed_dir, "final_model")
    prod_path = "./data/raw/dataset_kobe_prod.parquet"
    output_path = os.path.join(processed_dir, "resultado_producao.parquet")

    # Carregar modelo treinado
    logging.info("📦 Carregando o modelo treinado...")
    model = load_model(model_path)

    # Carregar dados de produção
    logging.info("📥 Carregando dados de produção...")
    df_prod = pd.read_parquet(prod_path)

    # Features esperadas pelo modelo
    expected_features = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']

    # Validar se todas as features necessárias estão presentes
    missing_features = [feature for feature in expected_features if feature not in df_prod.columns]
    if missing_features:
        raise ValueError(f"❌ As seguintes features estão ausentes nos dados de produção: {missing_features}")

    logging.info("✅ Todos os atributos necessários estão presentes.")

    # Remover a coluna 'shot_made_flag' se existir no dataset de produção
    if 'shot_made_flag' in df_prod.columns:
        df_prod = df_prod.drop(columns=['shot_made_flag'])

    # Realizar predições
    predictions = predict_model(model, data=df_prod)

    # Salvar resultados
    predictions.to_parquet(output_path, index=False)
    logging.info(f"📂 Resultados salvos em {output_path}")

if __name__ == "__main__":
    aplicar_modelo()
