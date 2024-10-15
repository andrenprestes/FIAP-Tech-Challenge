from flask import Flask, jsonify
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd
import requests
import numpy as np

app = Flask(__name__)
port = 5000
url_base_csv = 'http://vitibrasil.cnpuv.embrapa.br/download/'


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
        # Faz a requisição
        response = requests.get(url_base_csv + page + ".csv")

        # Verifica se o código de status HTTP é 200 (OK)
        if response.status_code == 200:
            # Converte o conteúdo da resposta em um objeto semelhante a um arquivo
            csv_data = StringIO(response.content.decode("utf-8"))
            df = pd.read_csv(csv_data, sep=";")
            df.fillna(0, inplace=True)

            if 'control' in df.columns:
                # Garante que a coluna 'control' seja do tipo string
                df['control'] = df['control'].astype(str)

                # Cria a máscara para verificar se os valores estão em maiúsculas
                mask_upper = df['control'].str.isupper()

                df_dict = {}
                indices = df[mask_upper].index.tolist()

                # Adiciona o final do DataFrame para capturar a última seção
                indices.append(len(df))

                if not indices or len(indices) == 1:
                    # Se não houver nenhuma linha em maiúsculas, retorna o DataFrame completo
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

# obter os dados de produção
@app.route('/producaoCSV')
def producao_csv():
    """
    Endpoint para obter dados de produção de uva.
    
    Returns:
        JSON: Retorna os dados do arquivo 'Producao.csv' em formato JSON.
    """
    return get_data("Producao")

# obter os dados de processamento por tipo
# opições: Viniferas, Americanas, Mesa e Semclass
@app.route('/processamentoCSV/tipo/<tipo>')
def processamento_csv(tipo):
    """
    Endpoint para obter dados de processamento de uva por tipo.
    
    Parameters:
        tipo (str): O tipo de processamento. Pode ser 'Viniferas', 'Americanas', 'Mesa', 'Semclass'.
    
    Returns:
        JSON: Retorna os dados do arquivo correspondente em formato JSON.
    """
    return get_data(f"Processa{tipo}")

# obter os dados de comercialização
@app.route('/comercializacaoCSV')
def comercializacao_csv():
    """
    Endpoint para obter dados de comercialização de uva.
    
    Returns:
        JSON: Retorna os dados do arquivo 'Comercio.csv' em formato JSON.
    """
    return get_data("Comercio")

# obter os dados de importaçao por tipo
# opições: Vinhos, Espumantes, Frescas, Passas e Suco
@app.route('/importacaoCSV/tipo/<tipo>')
def importacao_csv(tipo):
    """
    Endpoint para obter dados de importação de uva por tipo.
    
    Parameters:
        tipo (str): O tipo de importação. Pode ser 'Vinhos', 'Espumantes', 'Frescas', 'Passas', 'Suco'.
    
    Returns:
        JSON: Retorna os dados do arquivo correspondente em formato JSON.
    """
    return get_data(f"Imp{tipo}")

# obter os dados de exportação por tipo
# opições: Vinho, Espumantes, Uva e Suco
@app.route('/exportacaoCSV/tipo/<tipo>')
def exportacao_csv(tipo):
    """
    Endpoint para obter dados de exportação de uva por tipo.
    
    Parameters:
        tipo (str): O tipo de exportação. Pode ser 'Vinho', 'Espumantes', 'Uva', 'Suco'.
    
    Returns:
        JSON: Retorna os dados do arquivo correspondente em formato JSON.
    """
    return get_data(f"Exp{tipo}")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port)
