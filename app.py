from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask import Flask, jsonify, request
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd
import requests
import numpy as np


app = Flask(__name__)

# Configuração do JWT
app.config['JWT_SECRET_KEY'] = 'sua_chave_secreta_aqui'  # Defina uma chave secreta forte
jwt = JWTManager(app)

port = 5000
url_base_csv = 'http://vitibrasil.cnpuv.embrapa.br/download/'


# Função para criar token de acesso
@app.route('/login', methods=['POST'])
def login():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
    password = request.json.get('password', None)

    # Verificação simples de usuário/senha (substitua por sua lógica de verificação)
    if username != 'admin' or password != 'senha123':
        return jsonify({"msg": "Bad username or password"}), 401

    # Cria o token de acesso
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)


# Protege o endpoint usando JWT
@app.route('/producaoCSV')
@jwt_required()  # Requer autenticação JWT para acessar este endpoint
def producao_csv():
    """
    Endpoint para obter dados de produção de uva.
    
    Returns:
        JSON: Retorna os dados do arquivo 'Producao.csv' em formato JSON.
    """
    return get_data("Producao")


@app.route('/processamentoCSV/tipo/<tipo>')
@jwt_required()  # Requer autenticação JWT para acessar este endpoint
def processamento_csv(tipo):
    """
    Endpoint para obter dados de processamento de uva por tipo.
    
    Parameters:
        tipo (str): O tipo de processamento. Pode ser 'Viniferas', 'Americanas', 'Mesa', 'Semclass'.
    
    Returns:
        JSON: Retorna os dados do arquivo correspondente em formato JSON.
    """
    return get_data(f"Processa{tipo}")


@app.route('/comercializacaoCSV')
@jwt_required()  # Requer autenticação JWT para acessar este endpoint
def comercializacao_csv():
    """
    Endpoint para obter dados de comercialização de uva.
    
    Returns:
        JSON: Retorna os dados do arquivo 'Comercio.csv' em formato JSON.
    """
    return get_data("Comercio")


@app.route('/importacaoCSV/tipo/<tipo>')
@jwt_required()  # Requer autenticação JWT para acessar este endpoint
def importacao_csv(tipo):
    """
    Endpoint para obter dados de importação de uva por tipo.
    
    Parameters:
        tipo (str): O tipo de importação. Pode ser 'Vinhos', 'Espumantes', 'Frescas', 'Passas', 'Suco'.
    
    Returns:
        JSON: Retorna os dados do arquivo correspondente em formato JSON.
    """
    return get_data(f"Imp{tipo}")


@app.route('/exportacaoCSV/tipo/<tipo>')
@jwt_required()  # Requer autenticação JWT para acessar este endpoint
def exportacao_csv(tipo):
    """
    Endpoint para obter dados de exportação de uva por tipo.
    
    Parameters:
        tipo (str): O tipo de exportação. Pode ser 'Vinho', 'Espumantes', 'Uva', 'Suco'.
    
    Returns:
        JSON: Retorna os dados do arquivo correspondente em formato JSON.
    """
    return get_data(f"Exp{tipo}")


def get_data(page: str):
    """
    Faz uma requisição para a URL base com o nome da página fornecida e retorna os dados CSV processados como JSON.

    Parameters:
        page (str): O nome do arquivo CSV a ser baixado e processado (sem a extensão .csv).

    Returns:
        JSON: Retorna os dados em formato JSON, seja o DataFrame completo ou dividido em seções com base na coluna 'control'.
        Em caso de erro, retorna um código de status e uma mensagem de erro apropriada.
    """
    try:
        response = requests.get(url_base_csv + page + ".csv")
        if response.status_code == 200:
            csv_data = StringIO(response.content.decode("utf-8"))
            df = pd.read_csv(csv_data, sep=";")
            df.fillna(0, inplace=True)

            if 'control' in df.columns:
                df['control'] = df['control'].astype(str)
                mask_upper = df['control'].str.isupper()

                df_dict = {}
                indices = df[mask_upper].index.tolist()
                indices.append(len(df))

                if not indices or len(indices) == 1:
                    return jsonify(df.to_dict(orient='records'))

                for i in range(len(indices) - 1):
                    df_temp = df.iloc[indices[i]:indices[i + 1]]
                    key = df_temp.iloc[0]['control']
                    df_temp = df_temp.drop(columns=['control'])
                    df_dict[key] = df_temp.to_dict(orient='records')

                return jsonify(df_dict)
            else:
                return jsonify(df.to_dict(orient='records'))
        else:
            return jsonify(
                {"error": f"Failed to retrieve data, status code: {response.status_code}"}), response.status_code

    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port)
