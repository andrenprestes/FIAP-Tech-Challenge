import requests
import pandas as pd
from datetime import datetime, timezone
from s3_bucket_manager.manager import (
    save_dataframe_to_parquet,
    ensure_bucket_exists,
    load_parquet_from_s3
)

def get_historic(crypto_id="bitcoin", currency="usd", days=365):
    """
    Obtém o histórico de preços da criptomoeda utilizando a API do CoinGecko.

    Parâmetros:
        - crypto_id (str): O ID da criptomoeda (padrão: 'bitcoin').
        - currency (str): A moeda em que a criptomoeda será cotada (padrão: 'usd').
        - days (int): O número de dias históricos a serem obtidos (padrão: 365).

    Retorna:
        - DataFrame: Contendo os dados de preço e data da criptomoeda.
    """
    try:
        # Configurações da URL da API CoinGecko
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {"vs_currency": currency, "days": days}

        # Fazendo a requisição
        response = requests.get(url, params=params)
        response.raise_for_status()  # Levanta exceções para status HTTP 4xx/5xx

        # Processa os dados retornados
        data = response.json()

        # Verifica se a chave 'prices' está presente
        if "prices" not in data:
            raise ValueError("Dados de preços não encontrados na resposta da API.")

        prices = data["prices"]  # Lista de timestamps e preços
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

        return df
    except requests.exceptions.RequestException as e:
        print(f"Erro ao acessar a API do CoinGecko: {e}")
        return None
    except ValueError as e:
        print(f"Erro de dados: {e}")
        return None
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return None

def fill_db():
    """
    Preenche o banco de dados com os dados históricos de preços da criptomoeda.
    """
    bucket_name = "coinGecko_historic"
    
    # Obter dados históricos
    df = get_historic()
    
    if df is None:
        return {"msg": "Erro ao obter dados históricos da API CoinGecko."}, 500

    try:
        # Garantir que o bucket exista
        ensure_bucket_exists(bucket_name)

        # Salvar os dados no S3
        save_dataframe_to_parquet(df, bucket_name, s3_path="historic")
        print("Dados históricos processados e salvos no S3 com sucesso!")
        return {"msg": "Dados históricos processados e salvos no S3 com sucesso!"}, 200
    except Exception as e:
        print(f"Erro ao processar dados e salvar no S3: {e}")
        return {"msg": f"Erro ao processar dados e salvar no S3: {e}"}, 500

def get_curret_data():
    """
    Obtém a cotação atual da criptomoeda e atualiza o histórico no S3.
    """
    bucket_name = "coinGecko_historic"
    s3_path = "historic"

    try:
        # Carregar histórico salvo no S3
        df_historic = load_parquet_from_s3(bucket_name, s3_path)
        print("Histórico carregado do S3 com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar histórico do S3: {e}")
        return {"msg": f"Erro ao carregar histórico do S3: {e}"}, 500

    try:
        # Obter a cotação atual
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin",
            "vs_currencies": "usd"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()  # Levanta exceções para status HTTP 4xx/5xx

        data = response.json()

        if "bitcoin" in data and "usd" in data["bitcoin"]:
            current_price = data["bitcoin"]["usd"]
            current_timestamp = datetime.now(timezone.utc).timestamp() * 1000  # Converter para ms
            current_date = datetime.now(timezone.utc)

            # Criar novo DataFrame com a cotação atual
            new_data = pd.DataFrame([[current_timestamp, current_price, current_date]], 
                                    columns=["timestamp", "price", "date"])

            # Concatenar com o histórico
            df_historic = pd.concat([df_historic, new_data], ignore_index=True)

            # Salvar de volta no S3
            save_dataframe_to_parquet(df_historic, bucket_name, s3_path)
            print("Dados atualizados e salvos no S3 com sucesso!")
            return {"msg": "Dados atualizados e salvos no S3 com sucesso!"}, 200
        else:
            return {"msg": "Erro ao obter a cotação atual."}, 500

    except requests.exceptions.RequestException as e:
        print(f"Erro ao acessar a API do CoinGecko: {e}")
        return {"msg": f"Erro ao acessar a API do CoinGecko: {e}"}, 500
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return {"msg": f"Erro inesperado: {e}"}, 500
