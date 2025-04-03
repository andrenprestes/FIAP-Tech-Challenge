### Previsão de Preços do Bitcoin

#### **Descrição do Projeto**
Este projeto implementa um pipeline completo para a análise e previsão do preço do Bitcoin, utilizando dados históricos e técnicas de modelagem de séries temporais. O objetivo é fornecer previsões precisas e insights sobre o comportamento do mercado de criptomoedas.

---

#### **Tecnologias Utilizadas**
- **Python**: Linguagem de programação principal.
- **yfinance**: Biblioteca para obter dados históricos do Bitcoin diretamente da API do Yahoo Finance.
- **Flask**: Framework para criar uma API que facilita a interação com os dados e a execução de previsões.
- **Streamlit**: Biblioteca para desenvolver um dashboard interativo que permite a visualização das previsões.
- **ARIMA, Prophet, LSTM**: Modelos de previsão de séries temporais utilizados para análise.

---

#### **Estrutura do Projeto**
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

---

#### **Modelos de Previsão**
Os seguintes modelos foram testados para prever os preços do Bitcoin:
- **ARIMA**: Modelo adequado para séries temporais estacionárias.
- **LSTM (Long Short-Term Memory)**: Rede neural recorrente que captura dependências temporais complexas.
- **Prophet**: Modelo desenvolvido pelo Facebook, ideal para séries temporais com forte sazonalidade.

---

#### **Resultados**
O modelo LSTM apresentou o melhor desempenho em termos de precisão, seguido pelo ARIMA e, por último, o Prophet. As previsões são apresentadas em um dashboard interativo, permitindo que os usuários explorem as tendências de preços de forma intuitiva.

- [Acesse o dashboard](https://bitcoin-price-prediction-dashboard.streamlit.app/)
- [Acesse o repositório do dashboard](https://github.com/lis-r-barreto/bitcoin-price-predictor-app)

---

#### **Configuração do Ambiente Virtual (venv)**
1. **Criação do Ambiente Virtual**:
   No terminal Linux, execute o seguinte comando para criar um ambiente virtual:
   ```bash
   python3 -m venv venv
   ```

2. **Ativação do Ambiente Virtual**:
   Ative o ambiente virtual com o comando:
   ```bash
   source venv/bin/activate
   ```

3. **Instalação das Dependências**:
   Certifique-se de que o arquivo `requirements.txt` está no diretório do projeto. Em seguida, instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. **Desativação do Ambiente Virtual**:
   Após terminar o uso, desative o ambiente virtual com:
   ```bash
   deactivate
   ```

---

#### **Execução da Aplicação**
Para executar a aplicação, navegue até o diretório `fase_03`, onde o arquivo `app.py` está localizado. Use o seguinte comando no terminal:
```bash
cd fase_03
python3 app.py
```
Isso iniciará o servidor Flask, permitindo que você interaja com a API.

---

#### **Uso da API para Obter Preços de Bitcoin**
A API permite acessar informações sobre preços históricos e o valor atual do Bitcoin. Siga os passos abaixo para interagir com as rotas da API.

##### **1. Login para Obter o Token de Acesso**
Faça uma requisição POST para a rota `/login` com as credenciais de login. O token JWT retornado será usado para autenticação nas demais rotas.

Exemplo:
```bash
curl -X POST http://localhost:5000/login \
-H "Content-Type: application/json" \
-d '{"username": "admin", "password": "senha123"}'
```

**Resposta esperada**:
Um token JWT será retornado.

---

##### **2. Obter Preços Históricos do Bitcoin**
Use a rota `/historical` para acessar os preços históricos. Inclua o token JWT no cabeçalho da requisição.

Exemplo:
```bash
curl -X GET http://localhost:5000/historical \
-H "Authorization: Bearer <seu_token_jwt>"
```

---

##### **3. Obter o Valor Atual do Bitcoin**
Use a rota `/current_value` para acessar o valor atual do Bitcoin. Assim como na rota anterior, inclua o token JWT no cabeçalho.

Exemplo:
```bash
curl -X GET http://localhost:5000/current_value \
-H "Authorization: Bearer <seu_token_jwt>"
```

---

#### **Contribuições**
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

---

#### **Licença**
Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
