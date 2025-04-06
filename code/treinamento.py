import pandas as pd
import os
import mlflow
import joblib
from pycaret.classification import *
from sklearn.metrics import log_loss, f1_score
from mlflow.models.signature import infer_signature
from pycaret.classification import save_model


processed_dir = "./data/processed"
train_path = os.path.join(processed_dir, "base_train.parquet")
test_path = os.path.join(processed_dir, "base_test.parquet")

mlflow.set_experiment("Treinamento")

with mlflow.start_run(run_name="Treinamento"):

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df.drop(columns=['shot_made_flag'])
    y_train = train_df['shot_made_flag']
    X_test = test_df.drop(columns=['shot_made_flag'])
    y_test = test_df['shot_made_flag']


    clf1 = setup(
        data=train_df,
        target='shot_made_flag',
        session_id=123,
        verbose=False
    )

    # Treinamento - Regressão Logística
    log_model = create_model('lr')
    results_lr = pull()
    
    # Realiza predições e coleta a probabilidade de acerto
    predictions_lr = predict_model(log_model, data=test_df)
    
    if 'prediction_score' in predictions_lr.columns:
        prob_lr = predictions_lr['prediction_score']
        pred_labels_lr = predictions_lr['prediction_label']
    else:
        raise ValueError("A coluna 'prediction_score' não foi encontrada. Verifique o output do predict_model().")
    
    # Calculo das métricas
    log_loss_lr = log_loss(y_test, prob_lr)
    f1_score_lr = f1_score(y_test, pred_labels_lr)
    
    mlflow.log_metric("f1_score_lr", f1_score_lr)
    mlflow.log_metric("log_loss_lr", log_loss_lr)

    # Treinamento - Árvore de Decisão
    tree_model = create_model('dt')
    results_dt = pull()
    
    # Realiza predições e coleta a probabilidade de acerto
    predictions_dt = predict_model(tree_model, data=test_df)
    
    if 'prediction_score' in predictions_dt.columns:
        prob_dt = predictions_dt['prediction_score']
        pred_labels_dt = predictions_dt['prediction_label']
    else:
        raise ValueError("A coluna 'prediction_score' não foi encontrada. Verifique o output do predict_model().")
    
    # Calcular métricas
    log_loss_dt = log_loss(y_test, prob_dt)
    f1_score_dt = f1_score(y_test, pred_labels_dt)
    
    mlflow.log_metric("f1_score_dt", f1_score_dt)
    mlflow.log_metric("log_loss_dt", log_loss_dt)

    # Escolher o melhor modelo baseado no F1-Score
    if f1_score_dt > f1_score_lr:
        best_model = tree_model
        model_name = "DecisionTree"
    else:
        best_model = log_model
        model_name = "LogisticRegression"

    mlflow.log_param("final_model", model_name)
    mlflow.log_metric("best_model_f1_score", max(f1_score_dt, f1_score_lr))
    mlflow.log_metric("best_model_log_loss", log_loss_dt if f1_score_dt > f1_score_lr else log_loss_lr)
    

    input_example = train_df.drop(columns=["shot_made_flag"]).iloc[:5]
    signature = infer_signature(input_example, best_model.predict(input_example))

    # Salvar modelo final
    save_model(best_model, os.path.join(processed_dir, "final_model"))
    mlflow.sklearn.log_model(best_model, "final_model", input_example=input_example, signature=signature)

    print(f"Modelo registrado com sucesso no MLFlow com o Run ID: {mlflow.active_run().info.run_id}")
