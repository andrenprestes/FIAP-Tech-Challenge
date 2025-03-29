import yfinance as yt
import pandas as pd
from datetime import datetime, timezone

def get_historic(symbol="BTC-USD", period="1y"):
    """
    Obtém o histórico de preços da criptomoeda utilizando a API do Yahoo Finance.

    Parâmetros:
        - symbol (str): O símbolo da criptomoeda no Yahoo Finance (padrão: 'BTC-USD').
        - period (str): Período do histórico (padrão: '1y' para 1 ano).

    Retorna:
        - DataFrame: Contendo os dados de preço e data da criptomoeda.
    """
    try:
        df = yt.download(symbol, period=period, interval="1d")

        if df.empty:
            raise ValueError("Dados históricos não disponíveis.")

        df = df.reset_index()  # Converter índice para coluna normal
        df = df[["Date", "Adj Close"]].rename(columns={"Date": "date", "Adj Close": "price"})
        df["timestamp"] = df["date"].astype("int64") // 10**6  # Converter para timestamp em milissegundos

        return df
    except Exception as e:
        print(f"Erro ao obter dados históricos do Yahoo Finance: {e}")
        return None
    
print(get_historic())