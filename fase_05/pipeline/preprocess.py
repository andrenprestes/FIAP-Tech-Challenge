from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import unicodedata
import pandas as pd
import string
import nltk


def carrega_arquivos_json(caminho_prospects, caminho_vagas, caminho_applicants):
    """
    Lê os arquivos JSON e os converte em dicionários Python.

    Esta função foi separada para tornar o código mais modular e reutilizável,
    facilitando a leitura, testes e manutenções futuras. Ao encapsular a leitura
    dos arquivos em uma função específica, evitamos duplicação de código e garantimos
    que todos os arquivos sejam carregados de maneira consistente.

    Retorna:
        Tuple: dicionários (prospects, vagas, applicants)

    Espera-se que:
    - prospects.json contenha o relacionamento entre candidatos e vagas
    - vagas.json contenha os detalhes de cada vaga identificada por ID
    - applicants.json contenha os dados dos candidatos, também identificados por ID
    """
    with open(caminho_prospects, "r", encoding="utf-8") as f:
        prospects = json.load(f)
    with open(caminho_vagas, "r", encoding="utf-8") as f:
        vagas = json.load(f)
    with open(caminho_applicants, "r", encoding="utf-8") as f:
        applicants = json.load(f)
    return prospects, vagas, applicants


def consolidar_dados(prospects, vagas, applicants):
    """
    Essa função realiza a junção entre os dados das vagas, dos candidatos e do relacionamento entre eles (prospects).
    Cada registro representa uma tentativa de candidatura de um candidato a uma vaga específica.

    O campo 'contratado' é criado como variável alvo binária para o modelo supervisionado, assumindo valor 1
    se a situação do candidato for 'Contratado pela Decision' e 0 caso contrário.

    Isso é fundamental para treinar modelos de classificação que visam prever quais candidatos têm maior
    propensão a serem contratados, com base nas características extraídas.

    Retorna:
        Dataframe: Dados consolidados e cruzados.
    """
    registros = []

    for id_vaga, dados_vaga in prospects.items():
        prospects_vaga = dados_vaga.get("prospects", [])
        vaga_info = vagas.get(id_vaga, {})

        for prospect in prospects_vaga:
            cod_candidato = prospect.get("codigo")
            dados_candidato = applicants.get(cod_candidato, {})
            if not dados_candidato:
                continue

            contratado = int(prospect.get("situacao_candidado") == "Contratado pela Decision")

            registro = {
                "id_vaga": id_vaga,
                "id_candidato": cod_candidato,
                "situacao_candidado": prospect.get("situacao_candidado"),
                "recrutador": prospect.get("recrutador"),
                "data_candidatura": prospect.get("data_candidatura"),
                "comentario": prospect.get("comentario"),
                "contratado": contratado,
                # Dados da vaga
                "titulo_vaga": vaga_info.get("informacoes_basicas", {}).get("titulo_vaga"),
                "vaga_sap": vaga_info.get("informacoes_basicas", {}).get("vaga_sap"),
                "cliente": vaga_info.get("informacoes_basicas", {}).get("cliente"),
                "tipo_contratacao": vaga_info.get("informacoes_basicas", {}).get("tipo_contratacao"),
                "objetivo_vaga": vaga_info.get("informacoes_basicas", {}).get("objetivo_vaga"),
                "estado_vaga": vaga_info.get("perfil_vaga", {}).get("estado"),
                "cidade_vaga": vaga_info.get("perfil_vaga", {}).get("cidade"),
                "vaga_especifica_para_pcd": vaga_info.get("perfil_vaga", {}).get("vaga_especifica_para_pcd"),
                "faixa_etaria_vaga": vaga_info.get("perfil_vaga", {}).get("faixa_etaria"),
                "nivel_profissional": vaga_info.get("perfil_vaga", {}).get("nivel profissional"),
                "nivel_academico_vaga": vaga_info.get("perfil_vaga", {}).get("nivel_academico"),
                "ingles_vaga": vaga_info.get("perfil_vaga", {}).get("nivel_ingles"),
                "espanhol_vaga": vaga_info.get("perfil_vaga", {}).get("nivel_espanhol"),
                "outro_idioma_vaga": vaga_info.get("perfil_vaga", {}).get("outro_idioma"),
                "area_atuacao_vaga": vaga_info.get("perfil_vaga", {}).get("areas_atuacao"),
                "principais_atividades_vaga": vaga_info.get("perfil_vaga", {}).get("principais_atividades"),
                "competencia_tec_e_comp_vaga": vaga_info.get("perfil_vaga", {}).get(
                    "competencia_tecnicas_e_comportamentais"),
                "valor_venda": vaga_info.get("beneficios", {}).get("valor_venda"),
                "valor_compra_1": vaga_info.get("beneficios", {}).get("valor_compra_1"),
                "valor_compra_2": vaga_info.get("beneficios", {}).get("valor_compra_2"),
                # Dados do candidato
                "nome": dados_candidato.get("infos_basicas", {}).get("nome"),
                "pcd": dados_candidato.get("informacoes_pessoais", {}).get("pcd"),
                "email": dados_candidato.get("infos_basicas", {}).get("email"),
                "local_candidato": dados_candidato.get("infos_basicas", {}).get("local"),
                "objetivo_profissional": dados_candidato.get("infos_basicas", {}).get("objetivo_profissional"),
                "titulo_profissional": dados_candidato.get("informacoes_profissionais", {}).get("titulo_profissional"),
                "area_atuacao": dados_candidato.get("informacoes_profissionais", {}).get("area_atuacao"),
                "conhecimentos_tecnicos": dados_candidato.get("informacoes_profissionais", {}).get(
                    "conhecimentos_tecnicos"),
                "certificacoes": dados_candidato.get("informacoes_profissionais", {}).get("certificacoes"),
                "outras_certificacoes": dados_candidato.get("informacoes_profissionais", {}).get(
                    "outras_certificacoes"),
                "remuneracao": dados_candidato.get("informacoes_profissionais", {}).get("remuneracao"),
                "nivel_profissional": dados_candidato.get("informacoes_profissionais", {}).get("nivel_profissional"),
                "nivel_academico": dados_candidato.get("formacao_e_idiomas", {}).get("nivel_academico"),
                "nivel_ingles": dados_candidato.get("formacao_e_idiomas", {}).get("nivel_ingles"),
                "nivel_espanhol": dados_candidato.get("formacao_e_idiomas", {}).get("nivel_espanhol"),
                "outro_idioma": dados_candidato.get("formacao_e_idiomas", {}).get("outro_idioma"),
                "cargo_atual": dados_candidato.get("cargo_atual"),
                "cv": dados_candidato.get("cv_pt")
            }

            registros.append(registro)

    return pd.DataFrame(registros)


def pre_processar(df):
    nltk.download('stopwords')
    stopwords_pt = set(stopwords.words('portuguese'))

    def extrair_keywords_linha(atividades, competencias, n_top=20):
        """
        Extrai as principais palavras-chave de uma descrição de vaga (atividades + competências).

        Remove stopwords e retorna as `n_top` palavras mais frequentes da linha.

        Argumentos:
            atividades (str): texto de atividades da vaga
            competencias (str): texto de competências da vaga
            n_top (int): número de palavras mais frequentes a retornar

        Retorna:
            List[str]: lista de palavras-chave extraídas
        """
        texto = f"{atividades or ''} {competencias or ''}"
        texto = texto.lower()
        texto = re.sub(r'[^a-zà-ú0-9\s]', '', texto)
        tokens = texto.split()
        stopwords_pt = set(stopwords.words('portuguese'))
        stopwords_custom = {
            'vaga', 'atividades', 'responsabilidades', 'trabalhar', 'empresa',
            'experiência', 'profissional', 'atuar', 'área', 'conhecimento',
            'suporte', 'realizar', 'projetos', 'cliente', 'analista', 'tecnologia'
        }
        stopwords_total = stopwords_pt | stopwords_custom
        tokens = [t for t in tokens if t not in stopwords_total and len(t) > 2]
        mais_frequentes = Counter(tokens).most_common(n_top)
        keywords = [palavra for palavra, _ in mais_frequentes]
        return keywords


    def contar_keywords_cv_linha(cv, keywords):
        """
        Conta quantas palavras-chave aparecem no CV de cada candidato.
        """
        if not isinstance(cv, str):
            return 0
        cv = cv.lower()
        return sum(1 for kw in keywords if kw in cv)

    def calcular_similaridade_cv_atividade(df):
        """
        Conta quantas palavras-chave aparecem no texto do CV.

        Argumentos:
            cv (str): texto do currículo do candidato
            keywords (List[str]): lista de palavras-chave da vaga

        Retorna:
            int: quantidade de palavras-chave encontradas no CV
        """
        tfidf = TfidfVectorizer(max_features=500)

        textos_candidato = df['cv'].fillna("").astype(str)
        textos_vaga = df['principais_atividades_vaga'].fillna("").astype(str)

        tfidf_matrix = tfidf.fit_transform(pd.concat([textos_candidato, textos_vaga], ignore_index=True))
        tfidf_candidato = tfidf_matrix[:len(df)]
        tfidf_vaga = tfidf_matrix[len(df):]

        similaridades = [cosine_similarity(tfidf_candidato[i], tfidf_vaga[i])[0][0] for i in range(len(df))]

        df['match_cv_atividade'] = similaridades
        return df

    def match_texto_in_texto(base, alvo):
        """
        Verifica se a string `base` está contida na string `alvo`.

        Argumentos:
            base (str): texto do candidato
            alvo (str): texto da vaga

        Retorna:
            int: 1 se contido, 0 caso contrário
        """
        if isinstance(base, str) and isinstance(alvo, str):
            return int(base in alvo)
        return 0

    def normalizar_tipo_contratacao(texto):
        """
        Normaliza os tipos de contratação da coluna, tratando combinações, sinônimos,
        capitalização e ordem dos termos.
        """
        # 1. Tratamento inicial de valores nulos/vazios
        if pd.isna(texto) or not isinstance(texto, str) or str(texto).strip() == "":
            return "vazio"
        # 2. Converte para string, remove espaços e minúsculas
        texto = str(texto).strip().lower()
        # 3. Remover acentos
        texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
        # 4. Substituições e padronizações específicas
        texto = texto.replace('pj/autonomo', 'pj_autonomo')  # Unifica PJ/Autônomo
        texto = texto.replace('clt full', 'clt_full')  # Padroniza CLT Full
        texto = texto.replace('clt cotas', 'clt_cotas')  # Padroniza CLT Cotas
        texto = texto.replace('candidato podera escolher', '')  # Remove frase redundante
        texto = texto.replace('estagiario', 'estagio')  # Padroniza Estagiário
        # 5. Remover caracteres não alfanuméricos exceto underscores e espaços
        texto = re.sub(r'[^a-z0-9_\s]', '', texto)
        # 6. Compactar múltiplos espaços
        texto = re.sub(r'\s+', ' ', texto).strip()
        # 7. Separar os termos, remover vazios e normalizar a ordem para combinações
        # Ex: 'clt_full pj_autonomo' e 'pj_autonomo clt_full' viram a mesma lista e são ordenadas
        termos_individuais = sorted(list(set(filter(None, texto.split(' ')))))
        # Junta os termos novamente com '_' para formar a categoria combinada
        texto_normalizado = '_'.join(termos_individuais)
        # 8. Mapeamento final para categorias complexas (se necessário, pode ser mais genérico)
        # Essas regras são importantes para agrupar categorias que têm várias palavras
        # mas que você quer tratar como um único tipo.
        # A ordem é importante: das combinações mais longas para as mais curtas.
        if "clt_cotas_cooperado_estagio_hunting_pj_autonomo" in texto_normalizado:
            return "clt_cotas_cooperado_estagio_hunting_pj_autonomo"
        elif "clt_cotas_cooperado_estagio_pj_autonomo" in texto_normalizado:
            return "clt_cotas_cooperado_estagio_pj_autonomo"
        elif "clt_cotas_clt_full_cooperado_estagio_pj_autonomo" in texto_normalizado:
            return "clt_cotas_clt_full_cooperado_estagio_pj_autonomo"
        elif "clt_cotas_clt_full_cooperado_pj_autonomo" in texto_normalizado:
            return "clt_cotas_clt_full_cooperado_pj_autonomo"
        elif "clt_cotas_clt_full_pj_autonomo" in texto_normalizado:
            return "clt_cotas_clt_full_pj_autonomo"
        elif "clt_cotas_pj_autonomo" in texto_normalizado:
            return "clt_cotas_pj_autonomo"
        elif "clt_full_cooperado_pj_autonomo" in texto_normalizado:
            return "clt_full_cooperado_pj_autonomo"
        elif "clt_full_hunting_pj_autonomo" in texto_normalizado:
            return "clt_full_hunting_pj_autonomo"
        elif "cooperado_hunting_pj_autonomo" in texto_normalizado:
            return "cooperado_hunting_pj_autonomo"
        elif "cooperado_pj_autonomo" in texto_normalizado:
            return "cooperado_pj_autonomo"
        elif "clt_cotas_cooperado" in texto_normalizado:
            return "clt_cotas_cooperado"
        elif "clt_cotas_clt_full" in texto_normalizado:
            return "clt_cotas_clt_full"
        elif "clt_full_cooperado" in texto_normalizado:
            return "clt_full_cooperado"
        elif "clt_full_hunting" in texto_normalizado:
            return "clt_full_hunting"
        elif "clt_full_pj_autonomo" in texto_normalizado:
            return "clt_full_pj_autonomo"
        elif "pj_autonomo_hunting" in texto_normalizado:
            return "pj_autonomo_hunting"
        elif "clt_full" in texto_normalizado:
            return "clt_full"
        elif "pj_autonomo" in texto_normalizado:
            return "pj_autonomo"
        elif "hunting" in texto_normalizado:
            return "hunting"
        elif "cooperado" in texto_normalizado:
            return "cooperado"
        elif "clt_cotas" in texto_normalizado:
            return "clt_cotas"
        elif "estagio" in texto_normalizado:
            return "estagio"

        return texto_normalizado if texto_normalizado else "vazio"

    def limpa_texto(texto):
        """
        Realiza uma série de limpezas em um texto:
        1. Converte a entrada para string.
        2. Remove espaços em branco no início e fim.
        3. Converte para minúsculas.
        4. Remove acentos.
        5. Remove caracteres não alfanuméricos (mantém letras, números e espaços).
        6. Compacta múltiplos espaços em um único espaço.
        Args:
            texto (str ou qualquer tipo): O texto a ser limpo.
        Returns:
            str: O texto limpo, ou "vazio" se a entrada for inválida/nula.
        """
        if pd.isna(texto) or not isinstance(texto, str) or str(texto).strip() == "":
            return "vazio"  # Retorna "vazio" para NaN, não-strings ou strings vazias/apenas espaços
        # 1. Converte para string (garante que números, etc., sejam tratados como texto)
        texto = str(texto)
        # 2. Remove espaços em branco no início e fim e converte para minúsculas
        texto = texto.strip().lower()
        # 3. Remove acentos (normalização Unicode)
        texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
        # 4. Remove caracteres não alfanuméricos (mantém letras, números e espaços)
        texto = re.sub(r'[^a-z0-9\s,]', '', texto)
        # 5. Compacta múltiplos espaços em um único espaço
        texto = re.sub(r'\s+', ' ', texto).strip()  # .strip() final para pegar espaços extras criados por re.sub
        return texto

    colunas_texto = [
        "situacao_candidado", "recrutador", "comentario",
        "titulo_vaga", "vaga_sap", "cliente", "objetivo_vaga",
        "estado_vaga", "cidade_vaga", "vaga_especifica_para_pcd",
        "nivel_profissional", "nivel_academico_vaga", "ingles_vaga", "espanhol_vaga",
        "outro_idioma_vaga", "area_atuacao_vaga", "principais_atividades_vaga",
        "competencia_tec_e_comp_vaga", "nome", "pcd", "objetivo_profissional",
        "titulo_profissional", "area_atuacao", "conhecimentos_tecnicos",
        "certificacoes", "outras_certificacoes", "nivel_academico",
        "nivel_ingles", "nivel_espanhol", "outro_idioma", "cargo_atual", "cv"
    ]

    for col in colunas_texto:
        df[col] = df[col].apply(limpa_texto)

    # Features
    df["match_ingles"] = (df["nivel_ingles"] == df["ingles_vaga"]).astype(int)
    df["match_nivel_academico"] = (df["nivel_academico"] == df["nivel_academico_vaga"]).astype(int)
    df["match_area_atuacao"] = df.apply(
        lambda row: match_texto_in_texto(row["area_atuacao"], row["area_atuacao_vaga"]),
        axis=1
    )
    df["match_localidade"] = (df["local_candidato"] == df["cidade_vaga"]).astype(int)
    df["match_pcd"] = (df["pcd"] == df["vaga_especifica_para_pcd"]).astype(int)
    df['keywords_vaga'] = df.apply(
        lambda row: extrair_keywords_linha(row['principais_atividades_vaga'], row['competencia_tec_e_comp_vaga']),
        axis=1
    )
    df['match_qtd_keywords_cv'] = df.apply(
        lambda row: contar_keywords_cv_linha(row['cv'], row['keywords_vaga']),
        axis=1
    )
    calcular_similaridade_cv_atividade(df)
    # Aplica a função de normalização
    df['tipo_contratacao'] = df['tipo_contratacao'].apply(normalizar_tipo_contratacao)

    # handle_unknown='ignore' evita erros se surgir uma categoria nova que não foi vista no fit
    # DataFrame para armazenar as novas colunas codificadas
    df_encoded_features = pd.DataFrame()

    cols_to_encode = [
        "tipo_contratacao", "nivel_profissional", "nivel_academico",
        "nivel_ingles", "nivel_espanhol", "ingles_vaga", "espanhol_vaga",
        "nivel_academico_vaga"
    ]

    for col in cols_to_encode:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_data = ohe.fit_transform(df[[col]])
        # Ohe.get_feature_names_out() nos dá os nomes corretos para as novas colunas
        new_cols_names = ohe.get_feature_names_out([col])
        # Crie um DataFrame temporário para as colunas codificadas
        temp_df = pd.DataFrame(encoded_data, columns=new_cols_names, index=df.index)
        # Concatene as novas colunas ao DataFrame de features codificadas
        df_encoded_features = pd.concat([df_encoded_features, temp_df], axis=1)

    df_final = pd.concat([df, df_encoded_features], axis=1)
    df_final = df_final.drop(columns=cols_to_encode)

    return df_final


if __name__ == "__main__":
    # Caminhos locais dos arquivos (ajuste conforme necessário)
    path = "C:\\Users\\ffporto\\Desktop\\Estudo\\FIAP\\fase05\\data\\"
    caminho_prospects = f"{path}prospects.json"
    caminho_vagas = f"{path}vagas.json"
    caminho_applicants = f"{path}applicants.json"
    caminho_df_final = f"{path}dataset_processado.parquet"

    # Carregar e processar
    prospects, vagas, applicants = carrega_arquivos_json(caminho_prospects, caminho_vagas, caminho_applicants)
    df = consolidar_dados(prospects, vagas, applicants)
    df_final = pre_processar(df)
    df_final.to_parquet(caminho_df_final)
