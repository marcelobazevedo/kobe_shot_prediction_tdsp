import pandas as pd
import os
import mlflow
from pycaret.classification import load_model, predict_model
from sklearn.metrics import f1_score, log_loss
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aplicar_modelo():
    processed_dir = "./data/processed"
    model_path = os.path.join(processed_dir, "final_model")
    prod_path = "./data/raw/dataset_kobe_prod.parquet"
    output_path = os.path.join(processed_dir, "resultado_producao.parquet")

    mlflow.set_experiment("PipelineAplicacao")
    with mlflow.start_run(run_name="PipelineAplicacao"):


        logging.info("📦 Carregando o modelo treinado...")
        model = load_model(model_path)

        logging.info("📥 Carregando dados de produção...")
        df_prod = pd.read_parquet(prod_path)

        expected_features = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']


        missing_features = [feature for feature in expected_features if feature not in df_prod.columns]
        if missing_features:
            raise ValueError(f"As seguintes features estão ausentes nos dados de produção: {missing_features}")

        logging.info("Todos os atributos necessários estão presentes.")


        df_prod_filtered = df_prod[expected_features]

        predictions = predict_model(model, data=df_prod_filtered)

        df_prod['prediction_label'] = predictions['prediction_label']
        df_prod['prediction_score'] = predictions['prediction_score']

        df_prod.to_parquet(output_path, index=False)
        logging.info(f"Resultados salvos em {output_path}")

        if 'shot_made_flag' in df_prod.columns:
            y_true = df_prod['shot_made_flag']
            y_pred = predictions['prediction_label']
            y_proba = predictions['prediction_score']
            if y_true.isna().sum() > 0 or y_proba.isna().sum() > 0:
                logging.warning("Existem valores NaN nos dados. Removendo linhas inválidas.")
                valid_indices = y_true.notna() & y_proba.notna()
                y_true = y_true[valid_indices]
                y_proba = y_proba[valid_indices]
                y_pred = y_pred[valid_indices]
            try:
                log_loss_value = log_loss(y_true, y_proba)
                f1_score_value = f1_score(y_true, y_pred)
                
                mlflow.log_metric("log_loss", log_loss_value)
                mlflow.log_metric("f1_score", f1_score_value)

                logging.info(f"Log Loss registrado: {log_loss_value}")
                logging.info(f"F1 Score registrado: {f1_score_value}")
            except ValueError as e:
                logging.error(f"Erro ao calcular métricas: {str(e)}")
        else:
            logging.warning("Variável 'shot_made_flag' não encontrada nos dados de produção. Métricas não foram calculadas.")

if __name__ == "__main__":
    aplicar_modelo()

