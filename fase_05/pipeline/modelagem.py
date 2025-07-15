import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os

"""
Distribuição antes do SMOTE:
contratado
0    29988
1     1578
Name: count, dtype: int64

Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.98      0.97     12852
           1       0.38      0.27      0.31       677

    accuracy                           0.94     13529
   macro avg       0.67      0.62      0.64     13529
weighted avg       0.93      0.94      0.94     13529


Confusion Matrix:
[[12560   292]
 [  497   180]]
"""

def carregar_dados(path):
    return pd.read_parquet(path)

def extrair_features_completas(df):
    df.columns = df.columns.astype(str)
    # Vetorização textual combinada de CV, objetivo e atividades da vaga
    texto_completo = (
        df['cv'].fillna('') + ' ' +
        df['objetivo_profissional'].fillna('') + ' ' +
        df['titulo_profissional'].fillna('') + ' ' +
        df['principais_atividades_vaga'].fillna('')
    )

    tfidf = TfidfVectorizer(max_features=500)
    X_texto = tfidf.fit_transform(texto_completo)

    # Selecionar todas as colunas one-hot e numéricas, incluindo as de match
    X_estrut = df.filter(
        regex=r'^(tipo_contratacao_|nivel_profissional_|nivel_academico_|nivel_ingles_|nivel_espanhol_|ingles_vaga_|espanhol_vaga_|feature_mesma_cidade$|^match_)'
    ).reset_index(drop=True)

    # Concatenar tudo
    X_final = pd.concat([pd.DataFrame(X_texto.toarray()), X_estrut.reset_index(drop=True)], axis=1)
    return X_final, tfidf

def treinar_modelo_supervisionado(df):
    df.columns = df.columns.astype(str)
    X, tfidf = extrair_features_completas(df)
    X.columns = X.columns.astype(str)
    y = df['contratado']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Distribuição antes do SMOTE:")
    print(y_train.value_counts())
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # smote = SMOTE(random_state=42, sampling_strategy=0.5, k_neighbors=3)
    # X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # print("\nDistribuição após SMOTE:")
    # print(y_train_bal.value_counts())

    # scale_pos_weight = len(y_train_bal[y_train_bal == 0]) / len(y_train_bal[y_train_bal == 1])
    clf = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=14,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.5,
        colsample_bytree=0.8,
        random_state=42
    )
    # clf.fit(X_train_bal, y_train_bal)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    mlflow.set_experiment("modelo_candidato_sucesso")
    with mlflow.start_run():
        mlflow.log_params(clf.get_params())
        # Métricas adicionais
        mlflow.log_metric("acuracia", clf.score(X_test, y_test))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

        # Feature importance
        importances = clf.feature_importances_
        feature_names = X.columns.tolist()
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")

        # Input example e assinatura
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, clf.predict(X_test))
        mlflow.sklearn.log_model(clf, "modelo_xgboost", input_example=input_example, signature=signature)

    joblib.dump(clf, "modelo_xgboost.pkl")
    joblib.dump(tfidf, "vetorizador_tfidf.pkl")

    return clf

if __name__ == "__main__":
    path = "C:\\Users\\ffporto\\Desktop\\Estudo\\FIAP\\fase05\\data\\"
    df = carregar_dados(f"{path}dataset_processado.parquet")
    df.columns = df.columns.astype(str)
    clf = treinar_modelo_supervisionado(df)
    df.to_parquet(f"{path}dataset_clusterizado.parquet", index=False)
    print("Modelos treinados e salvos com sucesso!")
