import pandas as pd
import os
import mlflow
from pycaret.classification import *
from mlflow.models.signature import infer_signature

# Caminhos
processed_dir = "./data/processed"
train_path = os.path.join(processed_dir, "base_train.parquet")
test_path = os.path.join(processed_dir, "base_test.parquet")

# Nome da rodada
mlflow.set_experiment("Treinamento")

with mlflow.start_run(run_name="Treinamento"):

    # Carregando os dados
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Setup do PyCaret
    clf1 = setup(
        data=train_df,
        target='shot_made_flag',
        session_id=123,
        verbose=False
    )

    # Treinando Regressão Logística
    log_model = create_model('lr')
    results_lr = pull()
    f1_score_lr = results_lr.loc['Mean', 'F1']
    mlflow.log_metric("f1_score_lr", f1_score_lr)

    # Treinando Árvore de Decisão
    tree_model = create_model('dt')
    results_dt = pull()
    f1_score_dt = results_dt.loc['Mean', 'F1']
    mlflow.log_metric("f1_score_dt", f1_score_dt)

    # Escolher melhor modelo
    best_model = tree_model if f1_score_dt > f1_score_lr else log_model
    model_name = "DecisionTree" if best_model == tree_model else "LogisticRegression"
    mlflow.log_param("final_model", model_name)

    # Gerar input_example e assinatura do modelo
    input_example = train_df.drop(columns=["shot_made_flag"]).iloc[:5]
    signature = infer_signature(input_example, best_model.predict(input_example))

    # Salvar modelo final
    save_model(best_model, os.path.join(processed_dir, "final_model"))
    mlflow.sklearn.log_model(best_model, "final_model", input_example=input_example, signature=signature)
