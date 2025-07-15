from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier, cv, DMatrix
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd

"""
Melhor número de rounds: 42

Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.98      0.97     12852
           1       0.45      0.24      0.31       677

    accuracy                           0.95     13529
   macro avg       0.70      0.61      0.64     13529
weighted avg       0.93      0.95      0.94     13529


Confusion Matrix:
[[12653   199]
 [  517   160]]

ROC AUC: 0.8022099451958693
"""

def carregar_dados(path):
    return pd.read_parquet(path)


def extrair_features_completas(df):
    df.columns = df.columns.astype(str)
    texto_completo = (
            df['cv'].fillna('') + ' ' +
            df['objetivo_profissional'].fillna('') + ' ' +
            df['titulo_profissional'].fillna('') + ' ' +
            df['principais_atividades_vaga'].fillna('')
    )

    tfidf = TfidfVectorizer(max_features=100)
    X_texto = tfidf.fit_transform(texto_completo)

    X_estrut = df.filter(
        regex=r'^(tipo_contratacao_|nivel_profissional_|nivel_academico_|nivel_ingles_|nivel_espanhol_|ingles_vaga_|espanhol_vaga_|feature_mesma_cidade$|^match_|^qtd_keywords_cv$|^sim_cv_atividade$)'
    ).reset_index(drop=True)

    X_final = pd.concat([pd.DataFrame(X_texto.toarray()), X_estrut.reset_index(drop=True)], axis=1)
    return X_final, tfidf


def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    wtrain = dtrain.get_weight()
    wtest = dtest.get_weight()
    sum_weight = sum(wtrain) + sum(wtest)
    if sum(wtrain) > 0:
        wtrain *= sum_weight / sum(wtrain)
    if sum(wtest) > 0:
        wtest *= sum_weight / sum(wtest)
    dtrain.set_weight(wtrain)
    dtest.set_weight(wtest)
    return dtrain, dtest, param


def treinar_modelo_supervisionado(df):
    df.columns = df.columns.astype(str)
    X, tfidf = extrair_features_completas(df)
    X.columns = X.columns.astype(str)
    y = df['contratado']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Distribuição original:")
    print(y_train.value_counts())

    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)

    param = {
        'max_depth': 32,
        'learning_rate': 0.05,
        'max_delta_step': 10,
        'n_estimators': 300,
        'nthread': 16,
        'eta': 0.1,
        'subsample': 0.5,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 42
    }
    num_round = param['n_estimators']

    print("\nExecutando cross-validation com xgb.cv...")
    cv_results = cv(
        param,
        dtrain,
        num_boost_round=num_round,
        nfold=5,
        seed=42,
        metrics=['auc'],
        fpreproc=fpreproc,
        early_stopping_rounds=10,
        verbose_eval=10
    )

    best_num_round = len(cv_results)
    print(f"\nMelhor número de rounds: {best_num_round}")

    clf = XGBClassifier(
        max_depth=param['max_depth'],
        learning_rate=param['learning_rate'],
        max_delta_step=param['max_delta_step'],
        n_estimators=best_num_round,
        nthread=param['nthread'],
        eta=param['eta'],
        subsample=param['subsample'],
        colsample_bytree=param['colsample_bytree'],
        scale_pos_weight=float(np.sum(y_train == 0)) / np.sum(y_train == 1),
        objective=param['objective'],
        eval_metric=param['eval_metric'],
        use_label_encoder=False,
        random_state=42
    )
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
        mlflow.log_metric("acuracia", clf.score(X_test, y_test))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        importances = clf.feature_importances_
        feature_names = X.columns.tolist()
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")

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
