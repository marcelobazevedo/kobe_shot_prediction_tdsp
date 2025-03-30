import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Kobe Shot Prediction Dashboard", layout="wide")

st.title("🏀 Kobe Bryant Shot Prediction Dashboard")
st.sidebar.header("Filtros")

# Caminho do resultado
resultado_path = "./data/processed/resultado_producao.parquet"

if os.path.exists(resultado_path):
    df = pd.read_parquet(resultado_path)

    # Exibir amostra dos dados
    st.subheader("Amostra dos Dados")
    st.dataframe(df.sample(10), use_container_width=True)

    # Gráfico de dispersão dos pontos de arremesso
    st.subheader("Mapa de Arremessos")
    fig = px.scatter(
        df, x="lon", y="lat", color="prediction_label", 
        title="Distribuição dos Arremessos por Localização",
        labels={'lon': 'Longitude', 'lat': 'Latitude', 'prediction_label': 'Previsão'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Estatísticas gerais
    st.sidebar.subheader("Estatísticas Gerais")
    total_shots = len(df)
    successful_shots = df['prediction_label'].sum()
    accuracy = (successful_shots / total_shots) * 100

    st.sidebar.metric("Total de Arremessos", total_shots)
    st.sidebar.metric("Acertos Previstos", successful_shots)
    st.sidebar.metric("Taxa de Acerto (%)", round(accuracy, 2))

else:
    st.warning("Nenhum resultado de produção encontrado. Execute o script de aplicação primeiro.")
