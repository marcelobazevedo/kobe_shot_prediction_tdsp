import subprocess
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def executar_pipeline():
    try:
        logging.info("=== Etapa 1: Preparação dos Dados ===")
        subprocess.run(["python", "code/preparacao_dados.py"], check=True)
        logging.info("✅ Dados preparados com sucesso.")

        logging.info("=== Etapa 2: Treinamento do Modelo ===")
        subprocess.run(["python", "code/treinamento.py"], check=True)
        logging.info("✅ Modelos treinados com sucesso.")

        logging.info("=== Etapa 3: Aplicação em Produção ===")
        subprocess.run(["python", "code/aplicacao.py"], check=True)
        logging.info("✅ Aplicação concluída com sucesso.")

        logging.info("=== Etapa 4: Dashboard ===")
        logging.info("Inicie o dashboard com: streamlit run outputs/dashboards/streamlit_app.py")

        logging.info("🎉 Pipeline completo executado com sucesso!")

    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Erro na execução do pipeline: {e}")

if __name__ == "__main__":
    executar_pipeline()
