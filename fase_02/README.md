# Pipeline de Dados Batch Bovespa

## Descrição
Este projeto implementa um pipeline de dados automatizado para ingestão, processamento e análise de dados financeiros do índice Bovespa (B3). O pipeline foi projetado para ser escalável e utiliza serviços serverless da AWS. Ele abrange desde o web scraping no site oficial da B3 até a disponibilização dos dados processados no Amazon Athena para consultas analíticas.

## Arquitetura
A arquitetura do pipeline segue as etapas descritas abaixo:

1. **Extração de Dados**:
   - O script em Python utiliza Selenium para coletar os dados diários do pregão da B3 no site oficial.
   - Os dados são convertidos e armazenados em formato Parquet.

2. **Ingestão e Armazenamento no Amazon S3**:
   - Os dados brutos são enviados para um bucket no S3, particionados por data de processamento.

3. **Automação com AWS Lambda**:
   - Uma função Lambda é acionada automaticamente após a inserção de novos arquivos no bucket raw.
   - A Lambda inicia o job de ETL no AWS Glue.

4. **Transformação no AWS Glue**:
   - O Glue processa os dados aplicando as seguintes transformações:
     - Agrupamento e sumarização.
     - Cálculos com campos de data.
     - Geração de métricas analíticas.

5. **Armazenamento Refinado no S3**:
   - Os dados transformados são armazenados no bucket refined em formato Parquet, particionados por data e abreviação da ação.

6. **Catalogação Automática**:
   - Os dados processados são catalogados no AWS Glue Catalog.

7. **Disponibilização no Amazon Athena**:
   - Os dados ficam disponíveis no Athena para consultas SQL e análises.

![Arquitetura](https://github.com/user-attachments/assets/9f71013d-8e1f-45c6-a737-37dd82c9b005)

## Requisitos Atendidos

### Requisitos Obrigatórios
1. **Extração de Dados**:
   - Uso de Selenium para web scraping no site da B3.

2. **Ingestão no S3**:
   - Dados brutos armazenados em formato Parquet e particionados por data.

3. **Automação com Lambda**:
   - A função Lambda monitora o bucket raw e inicia automaticamente o job no AWS Glue.

4. **Transformações no AWS Glue**:
   - Realiza agrupamento, sumarização e cálculos baseados em data.
   - Gera métricas como frequência de transações e nível de participação.

5. **Armazenamento no Bucket Refined**:
   - Os dados processados são particionados por data e abreviação de ação.

6. **Catalogação no AWS Glue Catalog**:
   - Dados catalogados para serem consultados pelo Athena.

7. **Disponibilidade no Athena**:
   - Dados disponíveis para consultas SQL analíticas.

### Requisitos Opcionais
- **Notebooks no Athena**:
  - Exploração de dados com notebooks visuais no Athena.

## Tecnologias Utilizadas
- **AWS**:
  - Amazon S3
  - AWS Lambda
  - AWS Glue
  - AWS Glue Catalog
  - Amazon Athena
- **Python**:
  - Selenium para web scraping.
  - Pandas e PyArrow para processamento de dados.
  - Boto3 para interação com serviços AWS.

## Como Executar o Projeto

### 1. Configuração Inicial
- Configure sua conta AWS com permissões adequadas para S3, Lambda, Glue e Athena.
- Instale as dependências Python:
  ```bash
  pip install -r requirements.txt
  ```
- Defina as variáveis de ambiente:
  - `AWS_ACCESS_KEY_ID` e `AWS_SECRET_ACCESS_KEY` para autenticação.
  - Nome do bucket S3 para os dados brutos e refinados.

### 2. Execução do Script de Scrap
- Execute o script de scraping para baixar os dados:
  ```bash
  python app.py
  ```
- Rotas da API:
  - **/login**: Autenticação e geração de token.
    Exemplo:
    ```json
    {
      "username": "admin",
      "password": "senha123"
    }
    ```
  - **/bovespaDay**: Faz o download dos dados e envia para o bucket raw.

### 3. Processamento Automático
- O pipeline é acionado automaticamente:
  - A função Lambda inicia o job no Glue.
  - Dados processados são armazenados no bucket refined.

### 4. Consulta e Análise
- Use o Amazon Athena para consultar os dados transformados:
  - Acesse o console do Athena.
  - Execute consultas SQL para explorar os dados.

## Estrutura do Repositório
```plaintext
/
|-- README.md
|-- app.py
|-- scrap/
|   |-- __init__.py
|   |-- browser_detection.py
|   |-- scrap.py
|-- s3_bucket_manager/
|   |-- __init__.py
|   |-- manager.py
|-- lambda_handler/
|   |-- lambda_handler.py
|-- glue/
|   |-- transformations.sql
```

## Licença
Este projeto está licenciado sob a [MIT License](LICENSE).
