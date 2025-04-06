import streamlit as st
import pandas as pd
import os
import plotly.express as px
import mlflow
import joblib
from sklearn.metrics import log_loss
from mlflow.tracking import MlflowClient

st.set_page_config(page_title="Kobe Bryant Shot Prediction Dashboard", layout="wide")

st.title("Kobe Bryant Shot Prediction Dashboard")

model_path = "./data/processed/final_model.pkl"
resultado_path = "./data/processed/resultado_producao.parquet"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.warning("Modelo treinado não encontrado. Execute o treinamento primeiro.")

if os.path.exists(resultado_path):
    df = pd.read_parquet(resultado_path)
else:
    st.warning("Nenhum resultado de produção encontrado. Execute o script de aplicação primeiro.")

client = MlflowClient()

with st.sidebar:
    st.header("Simulação de Arremessos")

    lat = st.slider("Latitude", min_value=-250.0, max_value=250.0, value=0.0)
    lon = st.slider("Longitude", min_value=-300.0, max_value=300.0, value=0.0)
    minutes_remaining = st.slider("Minutos Restantes", min_value=0, max_value=12, value=5)
    period = st.selectbox("Período do Jogo", [1, 2, 3, 4])
    playoffs = st.selectbox("Playoffs", [0, 1])
    shot_distance = st.slider("Distância do Arremesso (pés)", min_value=0, max_value=50, value=10)

    if st.button("Fazer Previsão"):
        if model:
            input_data = pd.DataFrame({
                'lat': [lat],
                'lon': [lon],
                'minutes_remaining': [minutes_remaining],
                'period': [period],
                'playoffs': [playoffs],
                'shot_distance': [shot_distance]
            })

            try:
                prediction = model.predict(input_data)[0]
                prediction_score = model.predict_proba(input_data)[0][1]

                st.write("### Resultado da Previsão")
                st.write(f"Previsão: {'Acerto' if prediction == 1 else 'Erro'}")
                st.write(f"Probabilidade de Acerto: {prediction_score * 100:.2f}%")

                # Simular um valor verdadeiro para calcular o Log Loss (0 ou 1)
                true_value = st.selectbox("Resultado Real (Simulação)", [0, 1])

                # Calcular o Log Loss especificando os labels [0, 1]
                log_loss_value = log_loss([true_value], [[1 - prediction_score, prediction_score]], labels=[0, 1])
                st.write(f"Log Loss: {log_loss_value:.4f}")

                with mlflow.start_run(run_name="Simulacao_Arremesso"):
                    mlflow.log_param("lat", lat)
                    mlflow.log_param("lon", lon)
                    mlflow.log_param("minutes_remaining", minutes_remaining)
                    mlflow.log_param("period", period)
                    mlflow.log_param("playoffs", playoffs)
                    mlflow.log_param("shot_distance", shot_distance)
                    mlflow.log_metric("prediction_score", prediction_score)
                    mlflow.log_metric("prediction_label", prediction)
                    mlflow.log_metric("log_loss", log_loss_value)
            except ValueError as e:
                st.error(f"Erro ao fazer a previsão: {str(e)}")
        else:
            st.warning("O modelo não foi carregado corretamente.")

st.subheader("Mapa de Arremessos - Distribuição dos Arremessos por Localização")

if 'df' in locals():
    fig = px.scatter(
        df, x="lon", y="lat", color="prediction_label", 
        title="Distribuição dos Arremessos por Localização",
        labels={'lon': 'Longitude', 'lat': 'Latitude', 'prediction_label': 'Previsão'}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.sidebar.header("Painel de Métricas")
    total_shots = len(df)
    successful_shots = df[df['prediction_label'] == 1].shape[0]
    accuracy = (successful_shots / total_shots) * 100
    f1_score = (2 * successful_shots) / (total_shots + successful_shots) 

    st.sidebar.metric("Total de Arremessos", total_shots)
    st.sidebar.metric("Acertos Previstos", successful_shots)
    st.sidebar.metric("Taxa de Acerto (%)", round(accuracy, 2))
    st.sidebar.metric("F1 Score", round(f1_score, 2))
else:
    st.warning("Os dados de produção não foram carregados corretamente.")


st.subheader("Histórico de Simulações - Registradas no MLFlow")

runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"])

if runs:
    st.write("#### Últimas Simulações")
    df_runs = pd.DataFrame([
        {
            "Run ID": run.info.run_id,
            "Prediction Score": run.data.metrics.get("prediction_score", None),
            "Prediction Label": run.data.metrics.get("prediction_label", None),
            "Log Loss": run.data.metrics.get("log_loss", None),
            "Lat": run.data.params.get("lat", None),
            "Lon": run.data.params.get("lon", None),
            "Shot Distance": run.data.params.get("shot_distance", None)
        }
        for run in runs
    ])
    st.dataframe(df_runs, use_container_width=True)
else:
    st.write("Nenhuma simulação registrada encontrada.")

