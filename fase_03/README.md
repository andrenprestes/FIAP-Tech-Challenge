# Previsão de Preços do Bitcoin

Este projeto implementa um pipeline completo para a análise e previsão do preço do Bitcoin, utilizando dados históricos e técnicas de modelagem de séries temporais. O objetivo é fornecer previsões precisas e insights sobre o comportamento do mercado de criptomoedas.

## Tecnologias Utilizadas

- **Python**: Linguagem de programação principal.
- **yfinance**: Biblioteca para obter dados históricos do Bitcoin diretamente da API do Yahoo Finance.
- **Flask**: Framework para criar uma API que facilita a interação com os dados e a execução de previsões.
- **Streamlit**: Biblioteca para desenvolver um dashboard interativo que permite a visualização das previsões.
- **ARIMA, Prophet, LSTM**: Modelos de previsão de séries temporais utilizados para análise.

## Estrutura do Projeto

O repositório está organizado da seguinte forma:

```
FIAP-Tech-Challenge/
├── fase_03/
│   ├── README.md
│   ├── app.py
│   ├── data_handler.py
│   ├── data/
│   ├── notebooks/
│   │   ├── coingecko_365_projeto_fase3.ipynb
│   │   ├── yfinance_3k_projeto_fase3.ipynb
│   │   └── yfinance_get_amostra.ipynb
│   └── s3_bucket_manager/
│       └── manager.py
```

## Modelos de Previsão

Os seguintes modelos foram testados para prever os preços do Bitcoin:

- **ARIMA**: Modelo adequado para séries temporais estacionárias.
- **LSTM (Long Short-Term Memory)**: Rede neural recorrente que captura dependências temporais complexas.
- **Prophet**: Modelo desenvolvido pelo Facebook, ideal para séries temporais com forte sazonalidade.

## Resultados

O modelo LSTM apresentou o melhor desempenho em termos de precisão, seguido pelo ARIMA e, por último, o Prophet. As previsões são apresentadas em um dashboard interativo, permitindo que os usuários explorem as tendências de preços de forma intuitiva.

[Acesse o dashboard](https://bitcoin-price-prediction-dashboard.streamlit.app/)

[Acesse o repositório do dashboard](https://github.com/lis-r-barreto/bitcoin-price-predictor-app)

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
