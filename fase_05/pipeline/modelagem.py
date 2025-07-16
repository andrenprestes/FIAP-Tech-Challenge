from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, cv, DMatrix
from imblearn.combine import SMOTEENN
from mlflow.models.signature import infer_signature
from preprocess import pre_processar
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd


def fpreproc(dtrain, dtest, param):
    """
    Pré-processamento dos dados de treino e teste antes da validação cruzada do XGBoost.

    Esta função ajusta dinamicamente o parâmetro `scale_pos_weight` com base na proporção
    entre classes majoritária e minoritária. Além disso, ela reescala os pesos dos conjuntos
    de treino e teste para garantir que o impacto das instâncias seja proporcional e
    comparável durante o processo de validação cruzada.

    Args:
        dtrain (xgboost.DMatrix): Conjunto de dados de treino com rótulos e pesos.
        dtest (xgboost.DMatrix): Conjunto de dados de teste com rótulos e pesos.
        param (dict): Dicionário de parâmetros do modelo XGBoost.

    Returns:
        Tuple[xgboost.DMatrix, xgboost.DMatrix, dict]: Os objetos `dtrain` e `dtest`
        com pesos atualizados, e o dicionário de parâmetros com `scale_pos_weight` ajustado.
    """
    label = dtrain.get_label()
    # Check if there are instances of the minority class to avoid division by zero
    if np.sum(label == 1) > 0:
        ratio = float(np.sum(label == 0)) / np.sum(label == 1)
        param['scale_pos_weight'] = ratio
    else:
        # If no positive samples, scale_pos_weight might not be applicable or set to a default
        param['scale_pos_weight'] = 1.0 # Or handle as an error
        print("Warning: No positive samples found in dtrain for scale_pos_weight calculation.")

    wtrain = dtrain.get_weight()
    wtest = dtest.get_weight()

    # Only re-scale weights if they exist and sum > 0 to avoid division by zero
    if wtrain is not None and sum(wtrain) > 0:
        sum_weight_train = sum(wtrain)
    else:
        sum_weight_train = 0

    if wtest is not None and sum(wtest) > 0:
        sum_weight_test = sum(wtest)
    else:
        sum_weight_test = 0

    total_sum_weight = sum_weight_train + sum_weight_test

    if total_sum_weight > 0:
        if sum_weight_train > 0:
            wtrain *= total_sum_weight / sum_weight_train
        if sum_weight_test > 0:
            wtest *= total_sum_weight / sum_weight_test

        if wtrain is not None:
            dtrain.set_weight(wtrain)
        if wtest is not None:
            dtest.set_weight(wtest)
    return dtrain, dtest, param


def criar_coluna_contratado_refinada(df):
    """
    Refina a coluna 'contratado' com base na situação do candidato e separa o dataset
    entre dados de treinamento (com rótulo definido) e dados em andamento (sem rótulo).

    Define `contratado = 1` para situações claramente bem-sucedidas no processo seletivo,
    `contratado = 0` para rejeições ou desistências, e mantém como NaN os casos indefinidos
    ou em andamento. Após o processamento, separa o DataFrame original em dois subconjuntos:
    um para treinamento supervisionado e outro com candidatos ainda em processo.

    Args:
        df (pd.DataFrame): DataFrame contendo, entre outras, a coluna 'situacao_candidado'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_treinamento: Subconjunto com a coluna 'contratado' preenchida (0 ou 1), pronto para treinamento.
            - df_em_andamento: Subconjunto com candidatos sem definição final, sem a coluna 'contratado'.
    """
    contratado_status = [
        'contratado pela decision',
        'contratado como hunting',
        'proposta aceita'
    ]
    nao_contratado_status = [
        'nao aprovado pelo cliente',
        'desistiu',
        'nao aprovado pelo rh',
        'nao aprovado pelo requisitante',
        'sem interesse nesta vaga',
        'desistiu da contratacao',
        'recusado'
    ]

    df['contratado'] = np.nan

    df.loc[df['situacao_candidado'].isin(contratado_status), 'contratado'] = 1
    df.loc[df['situacao_candidado'].isin(nao_contratado_status), 'contratado'] = 0

    df_treinamento = df.dropna(subset=['contratado']).copy()
    df_treinamento['contratado'] = df_treinamento['contratado'].astype(int)

    # df_em_andamento now explicitly contains only rows where 'contratado' was NaN,
    # and the 'contratado' column is dropped as it's not applicable for prediction
    df_em_andamento = df[df['contratado'].isna()].copy()
    if 'contratado' in df_em_andamento.columns:
        df_em_andamento.drop(columns=['contratado'], inplace=True)

    return df_treinamento, df_em_andamento


def carregar_dados(path):
    """
    Carrega dados a partir de um arquivo no formato Parquet.

    Args:
        path (str): Caminho completo para o arquivo .parquet.

    Returns:
        pd.DataFrame: DataFrame contendo os dados carregados do arquivo.
    """
    return pd.read_parquet(path)


def extrair_e_transformar_features(df_input, tfidf_model=None, ohe_models=None, original_feature_columns=None, is_training=True):
    """
    Extrai e transforma um conjunto completo de features a partir de dados textuais e estruturados,
    aplicando vetorização TF-IDF e One-Hot Encoding.

    Args:
        df_input (pd.DataFrame): DataFrame contendo as colunas de texto e estruturadas.
        tfidf_model (TfidfVectorizer, opcional): Modelo TF-IDF previamente ajustado.
            Se None e is_training for True, um novo modelo será ajustado.
        ohe_models (dict, opcional): Dicionário de objetos OneHotEncoder previamente ajustados.
            As chaves são os nomes das colunas e os valores são os objetos OHE.
            Se None e is_training for True, novos modelos serão ajustados.
        original_feature_columns (list, opcional): Lista de nomes das colunas de features esperadas
            para garantir consistência na ordem e presença das colunas.
        is_training (bool): Indica se a função está sendo chamada para o conjunto de treino (True)
            ou para predição/teste (False).

    Returns:
        Tuple[pd.DataFrame, TfidfVectorizer, dict, list]:
            - Um DataFrame com as features combinadas (TF-IDF + One-Hot + numéricas/binárias).
            - O objeto `TfidfVectorizer` (ajustado ou passado).
            - O dicionário de objetos `OneHotEncoder` (ajustados ou passados).
            - A lista de nomes das colunas finais de features.
    """
    df = df_input.copy()
    df.columns = df.columns.astype(str)

    # 1. Pré-processamento de texto e features diretas (comum a treino e inferência)
    # A função pre_processar agora faz isso e não mais o One-Hot Encoding.
    df = pre_processar(df)


    texto_completo = (
            df['cv'].fillna('') + ' ' +
            df['objetivo_profissional'].fillna('') + ' ' +
            df['titulo_profissional'].fillna('') + ' ' +
            df['principais_atividades_vaga'].fillna('')
    )

    # TF-IDF
    if is_training:
        tfidf = TfidfVectorizer(max_features=100)
        X_texto = tfidf.fit_transform(texto_completo)
    else:
        if tfidf_model is None:
            raise ValueError("tfidf_model must be provided for prediction/test.")
        tfidf = tfidf_model
        X_texto = tfidf.transform(texto_completo)

    X_texto_df = pd.DataFrame(X_texto.toarray())
    # Ensure TF-IDF column names are strings
    X_texto_df.columns = [f'tfidf_{i}' for i in range(X_texto_df.shape[1])]


    # One-Hot Encoding
    cols_to_encode = [
        "tipo_contratacao", "nivel_profissional", "nivel_academico",
        "nivel_ingles", "nivel_espanhol", "ingles_vaga", "espanhol_vaga",
        "nivel_academico_vaga"
    ]

    ohe_fitted_models = {}
    df_encoded_features = pd.DataFrame(index=df.index) # Initialize with original index

    for col in cols_to_encode:
        if is_training:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = ohe.fit_transform(df[[col]])
            ohe_fitted_models[col] = ohe # Save the fitted OHE model
        else:
            if ohe_models is None or col not in ohe_models:
                raise ValueError(f"OneHotEncoder for column '{col}' must be provided for prediction/test.")
            ohe = ohe_models[col]
            encoded_data = ohe.transform(df[[col]])

        new_cols_names = ohe.get_feature_names_out([col])
        temp_df = pd.DataFrame(encoded_data, columns=new_cols_names, index=df.index)
        df_encoded_features = pd.concat([df_encoded_features, temp_df], axis=1)

    # Numeric and binary features (already processed by pre_processar)
    X_numeric_binary = df.filter(
        regex=r'^(match_ingles|match_nivel_academico|match_area_atuacao|match_localidade|match_pcd|qtd_keywords_cv|match_cv_atividade$)'
    ).reset_index(drop=True)

    # Combine all features
    X_final = pd.concat([X_texto_df, df_encoded_features.reset_index(drop=True), X_numeric_binary], axis=1)
    X_final.columns = X_final.columns.astype(str)

    if is_training:
        # For training, capture the final column names
        final_feature_columns = X_final.columns.tolist()
    else:
        # For prediction, reindex to match the training columns, filling missing with 0
        if original_feature_columns is None:
            raise ValueError("original_feature_columns must be provided for prediction/test.")
        X_final = X_final.reindex(columns=original_feature_columns, fill_value=0)
        final_feature_columns = original_feature_columns # Just to return consistently

    return X_final, tfidf, ohe_fitted_models, final_feature_columns


def treinar_modelo_supervisionado(df_treinamento_input):
    """
    Treina um modelo supervisionado XGBoost com balanceamento de classes e clustering como feature adicional.

    Esta função realiza o pré-processamento completo dos dados, incluindo:
    - Extração de features TF-IDF e estruturadas (com One-Hot Encoding).
    - Balanceamento das classes com SMOTEENN (over + under sampling).
    - Ajuste dos hiperparâmetros via cross-validation com `xgb.cv`.
    - Treinamento final com o número ideal de árvores (`n_estimators`).
    - Avaliação com métricas de classificação e registro no MLflow.
    - Salvamento do modelo, do vetor TF-IDF e dos OneHotEncoders via `joblib`.

    Args:
        df_treinamento_input (pd.DataFrame): DataFrame contendo os dados de treino,
            já rotulados com a coluna 'contratado'.

    Returns:
        Tuple[XGBClassifier, TfidfVectorizer, dict, List[str]]:
            - O modelo XGBoost treinado.
            - O vetorizador TF-IDF ajustado.
            - O dicionário de OneHotEncoders ajustados.
            - A lista de nomes das colunas finais de features.
    """
    df_treinamento_input.columns = df_treinamento_input.columns.astype(str)

    # Extrai e transforma features no conjunto de TREINO.
    # tfidf e ohe_models serão ajustados aqui.
    X, tfidf_model, ohe_models, original_feature_columns = extrair_e_transformar_features(
        df_treinamento_input, is_training=True
    )

    y = df_treinamento_input['contratado']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Distribuição original do treinamento:")
    print(y_train.value_counts())

    # Apply SMOTEENN on the training set
    smoteenn = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smoteenn.fit_resample(X_train, y_train)

    print("\nDistribuição após SMOTEENN no treinamento:")
    print(y_train_res.value_counts())

    # Prepare DMatrix for xgb.cv with the resampled data
    dtrain = DMatrix(X_train_res, label=y_train_res)
    # X_test needs to be transformed using the fitted TF-IDF and OHE models.
    # For now, X_test already is, as it's a split of X which was generated with the fitted models.
    # However, if we were loading X_test from an external source, it would need the full transformation.
    # Let's create DMatrix for X_test as well.
    dtest = DMatrix(X_test, label=y_test)

    param = {
        'max_depth': 8,
        'learning_rate': 0.05,
        'max_delta_step': 1,
        'nthread': 16,
        'subsample': 0.5,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 42
    }
    num_round = 300  # Max rounds for CV

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

    best_num_round = len(cv_results) if len(cv_results) > 0 else num_round
    print(f"\nMelhor número de rounds: {best_num_round}")

    clf = XGBClassifier(
        max_depth=param['max_depth'],
        learning_rate=param['learning_rate'],
        max_delta_step=param['max_delta_step'],
        n_estimators=best_num_round,
        nthread=param['nthread'],
        subsample=param['subsample'],
        colsample_bytree=param['colsample_bytree'],
        # scale_pos_weight is calculated based on the RESAMPLED data for the final model fit
        scale_pos_weight=float(np.sum(y_train_res == 0)) / np.sum(y_train_res == 1),
        objective=param['objective'],
        eval_metric=param['eval_metric'],
        use_label_encoder=False,
        random_state=42
    )
    # Fit the model on the RESAMPLED training data!
    clf.fit(X_train_res, y_train_res)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    print("\nClassification Report (Padrão 0.5):")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix (Padrão 0.5):")
    print(confusion_matrix(y_test, y_pred))
    print("\nROC AUC:", roc_auc_score(y_test, y_pred_proba))

    mlflow.set_experiment("modelo_candidato_sucesso")
    with mlflow.start_run():
        mlflow.log_params(clf.get_params())
        mlflow.log_metric("acuracia", clf.score(X_test, y_test))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        mlflow.log_metric("precision_class1", precision_score(y_test, y_pred, pos_label=1))
        mlflow.log_metric("recall_class1", recall_score(y_test, y_pred, pos_label=1))
        mlflow.log_metric("f1_score_class1", f1_score(y_test, y_pred, pos_label=1))

        importances = clf.feature_importances_
        feature_names = original_feature_columns
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")

        # Use the raw X_test (before DMatrix conversion if needed) as input_example
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, clf.predict(X_test))
        mlflow.sklearn.log_model(clf, "modelo_xgboost", input_example=input_example, signature=signature)

    joblib.dump(clf, "modelo_xgboost.pkl")
    joblib.dump(tfidf_model, "vetorizador_tfidf.pkl")  # Save the fitted tfidf vectorizer
    joblib.dump(ohe_models, "one_hot_encoders.pkl") # Save the dictionary of fitted OHE models
    joblib.dump(original_feature_columns, "feature_columns.pkl") # Save the list of feature names

    return clf, tfidf_model, ohe_models, original_feature_columns


if __name__ == "__main__":
    path = "C:\\Users\\ffporto\\Desktop\\Estudo\\FIAP\\fase05\\data\\"
    # Ensure dataset_processado.parquet is generated by the new preprocess.py first
    df = carregar_dados(f"{path}dataset_processado.parquet")
    df.columns = df.columns.astype(str)

    # 1. Prepare training data and "in-progress" data
    # This separation happens on the raw, pre-processed (text/direct features) dataframe
    df_treinamento, df_em_andamento = criar_coluna_contratado_refinada(df)

    # 2. Train the model and get all fitted transformers and feature columns
    clf, tfidf_model, ohe_models, original_feature_columns = treinar_modelo_supervisionado(df_treinamento)

    # 3. Prepare "in-progress" data for prediction using the loaded/fitted transformers
    X_em_andamento, _, _, _ = extrair_e_transformar_features(
        df_em_andamento,
        tfidf_model=tfidf_model,
        ohe_models=ohe_models,
        original_feature_columns=original_feature_columns,
        is_training=False # IMPORTANT: Set to False for prediction/test
    )

    # 4. Make probability predictions for "in-progress" candidates
    probabilities_em_andamento = clf.predict_proba(X_em_andamento)[:, 1]

    # 5. Add predictions back to the 'df_em_andamento' DataFrame
    df_em_andamento['prob_contratado'] = probabilities_em_andamento

    # 6. Classify with an adjusted threshold for actionable insights
    # Adjust this threshold based on your desired balance of precision and recall for 'contratado'
    threshold_predicao = 0.5  # Example: You might want to experiment with this value
    df_em_andamento['predicao_contratado'] = (df_em_andamento['prob_contratado'] > threshold_predicao).astype(int)

    print("\n--- Candidatos Em Andamento com Previsões ---")
    # Display top 10 candidates with highest probability of being hired
    print(df_em_andamento[['situacao_candidado', 'prob_contratado', 'predicao_contratado']].sort_values(
        by='prob_contratado', ascending=False).head(10))

    # You can save this DataFrame with predictions for further analysis
    df_em_andamento.to_parquet(f"{path}dataset_em_andamento_com_predicao.parquet", index=False)
    print("\nModelos treinados e salvos com sucesso! Previsões para candidatos em andamento geradas.")