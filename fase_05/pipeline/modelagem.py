from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier, cv, DMatrix
from imblearn.combine import SMOTEENN
from mlflow.models.signature import infer_signature
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


def extrair_features_completas(df):
    """
    Extrai um conjunto completo de features a partir de dados textuais e estruturados,
    aplicando vetorização TF-IDF e selecionando colunas específicas para modelagem supervisionada.

    Essa função deve ser usada apenas na etapa de preparação do conjunto de treino, pois
    realiza o ajuste (`fit`) do TfidfVectorizer com base no conteúdo textual.

    Args:
        df (pd.DataFrame): DataFrame contendo as colunas de texto (como 'cv', 'objetivo_profissional')
            e colunas estruturadas codificadas por One-Hot ou binárias.

    Returns:
        Tuple[pd.DataFrame, TfidfVectorizer]:
            - Um DataFrame com as features combinadas (TF-IDF + colunas estruturadas).
            - O objeto `TfidfVectorizer` já ajustado, que poderá ser reutilizado na etapa de predição.
    """
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
    # Ensure column names are strings for XGBoost compatibility
    X_final.columns = X_final.columns.astype(str)
    return X_final, tfidf


def extrair_features_para_predicao(df_para_prever, tfidf_model, original_feature_columns):
    """
    Extrai as features necessárias para predição, garantindo consistência com o conjunto de treino.

    Essa função aplica a transformação TF-IDF em novos dados utilizando um modelo já treinado,
    e concatena com as features estruturadas do DataFrame. Também garante que a matriz final
    tenha exatamente as mesmas colunas que o conjunto de treino, preenchendo com zero as
    colunas ausentes se necessário.

    Args:
        df_para_prever (pd.DataFrame): DataFrame com os dados dos candidatos para predição.
        tfidf_model (TfidfVectorizer): Modelo TF-IDF previamente ajustado no conjunto de treino.
        original_feature_columns (List[str]): Lista de nomes das colunas de features usadas no treino,
            usada para reindexar e alinhar as colunas de entrada do modelo.

    Returns:
        pd.DataFrame: DataFrame com as mesmas features (em ordem e formato) utilizadas durante o treinamento.
    """
    df_para_prever.columns = df_para_prever.columns.astype(str)
    texto_completo = (
            df_para_prever['cv'].fillna('') + ' ' +
            df_para_prever['objetivo_profissional'].fillna('') + ' ' +
            df_para_prever['titulo_profissional'].fillna('') + ' ' +
            df_para_prever['principais_atividades_vaga'].fillna('')
    )

    # Use the tfidf_model (already fitted) to transform the new data
    X_texto = tfidf_model.transform(texto_completo)

    X_estrut = df_para_prever.filter(
        regex=r'^(tipo_contratacao_|nivel_profissional_|nivel_academico_|nivel_ingles_|nivel_espanhol_|ingles_vaga_|espanhol_vaga_|feature_mesma_cidade$|^match_|^qtd_keywords_cv$|^sim_cv_atividade$)'
    ).reset_index(drop=True)

    X_final = pd.concat([pd.DataFrame(X_texto.toarray()), X_estrut.reset_index(drop=True)], axis=1)
    X_final.columns = X_final.columns.astype(str)  # Ensure column names are strings

    # Reindex to ensure all columns from training are present, filling missing with 0 (for new TF-IDF features not seen)
    # This is important if max_features in TfidfVectorizer leads to different column counts
    X_final = X_final.reindex(columns=original_feature_columns, fill_value=0)
    return X_final


def treinar_modelo_supervisionado(df_treinamento_input):
    """
    Treina um modelo supervisionado XGBoost com balanceamento de classes e clustering como feature adicional.

    Esta função realiza o pré-processamento completo dos dados, incluindo:
    - Extração de features TF-IDF e estruturadas.
    - Balanceamento das classes com SMOTEENN (over + under sampling).
    - Ajuste dos hiperparâmetros via cross-validation com `xgb.cv`.
    - Treinamento final com o número ideal de árvores (`n_estimators`).
    - Avaliação com métricas de classificação e registro no MLflow.
    - Salvamento do modelo e do vetor TF-IDF via `joblib`.

    Args:
        df_treinamento_input (pd.DataFrame): DataFrame contendo os dados de treino,
            já rotulados com a coluna 'contratado'.

    Returns:
        Tuple[XGBClassifier, TfidfVectorizer, List[str]]:
            - O modelo XGBoost treinado.
            - O vetorizador TF-IDF ajustado.
            - A lista de nomes das colunas finais de features (incluindo as de cluster).
    """
    df_treinamento_input.columns = df_treinamento_input.columns.astype(str)
    X, tfidf = extrair_features_completas(df_treinamento_input)  # `tfidf` is now returned
    X.columns = X.columns.astype(str)
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
    # IMPORTANT: Do not apply fpreproc's scale_pos_weight if SMOTEENN already balanced the data to 1:1
    # If the ratio is very close to 1 after SMOTEENN, scale_pos_weight in fpreproc will be ~1.
    # We can pass an empty dict for param to fpreproc if we explicitly don't want scale_pos_weight applied here.
    dtrain = DMatrix(X_train_res, label=y_train_res)
    dtest = DMatrix(X_test, label=y_test)  # X_test and y_test remain unresampled

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
        fpreproc=fpreproc,  # Comment out or remove if SMOTEENN fully balances for CV
        # If SMOTEENN creates close to 1:1 ratio, scale_pos_weight will be near 1, effectively doing nothing.
        # Keeping it might still be fine, but often omitted if explicit resampling is done.
        early_stopping_rounds=10,
        verbose_eval=10
    )

    best_num_round = len(cv_results) if len(
        cv_results) > 0 else num_round  # Fallback in case early stopping doesn't trigger
    print(f"\nMelhor número de rounds: {best_num_round}")

    clf = XGBClassifier(
        max_depth=param['max_depth'],
        learning_rate=param['learning_rate'],
        max_delta_step=param['max_delta_step'],
        n_estimators=best_num_round,
        nthread=param['nthread'],
        subsample=param['subsample'],
        colsample_bytree=param['colsample_bytree'],
        # If SMOTEENN fully balances, scale_pos_weight should be 1 or omitted.
        # float(np.sum(y_train_res == 0)) / np.sum(y_train_res == 1) will be 1 if perfectly balanced.
        scale_pos_weight=float(np.sum(y_train_res == 0)) / np.sum(y_train_res == 1),  # Recalculate for resampled data
        objective=param['objective'],
        eval_metric=param['eval_metric'],
        use_label_encoder=False,
        random_state=42
    )
    # Fit the model on the RESAMPLED training data!
    clf.fit(X_train_res, y_train_res)  # <-- THIS WAS THE KEY CHANGE HERE

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
        feature_names = X.columns.tolist()  # Get feature names from the original X before resampling
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")

        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, clf.predict(X_test))
        mlflow.sklearn.log_model(clf, "modelo_xgboost", input_example=input_example, signature=signature)

    joblib.dump(clf, "modelo_xgboost.pkl")
    joblib.dump(tfidf, "vetorizador_tfidf.pkl")  # Save the fitted tfidf vectorizer

    return clf, tfidf, X.columns.tolist()  # Return clf, tfidf, and the list of feature columns


if __name__ == "__main__":
    path = "C:\\Users\\ffporto\\Desktop\\Estudo\\FIAP\\fase05\\data\\"
    df = carregar_dados(f"{path}dataset_processado.parquet")
    df.columns = df.columns.astype(str)

    # 1. Prepare training data and "in-progress" data
    df_treinamento, df_em_andamento = criar_coluna_contratado_refinada(df)

    # 2. Train the model and get the TF-IDF vectorizer and original feature columns
    clf, tfidf_model, original_feature_columns = treinar_modelo_supervisionado(df_treinamento)

    # 3. Prepare "in-progress" data for prediction using the loaded tfidf_model
    # This new `extrair_features_para_predicao` function will ensure consistency
    X_em_andamento = extrair_features_para_predicao(df_em_andamento, tfidf_model, original_feature_columns)

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