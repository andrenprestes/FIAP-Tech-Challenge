# Pipeline de Dados Batch Bovespa

## Descrição
Este projeto tem como objetivo construir um pipeline de dados completo para a ingestão, processamento e análise de dados da Bovespa (B3) utilizando serviços serverless da AWS. O pipeline realiza desde a extração dos dados no site oficial da B3 até a disponibilização de dados processados e analisáveis no Amazon Athena.

## Arquitetura
A arquitetura do pipeline segue o seguinte fluxo:

1. **Scrap de dados do site oficial da B3**:
   - Um script Python realiza a extração de dados do pregão da B3 e os salva em formato Parquet.
   - Os dados são particionados diariamente.

2. **AWS Cloud Pipeline**:
   - **Raw Bucket (S3)**: Os dados extraídos são ingeridos no Amazon S3.
   - **Lambda Trigger**: Uma função Lambda é acionada automaticamente após a ingestão dos dados no bucket, iniciando um job no AWS Glue.
   - **AWS Glue (ETL)**: O job Glue realiza as transformações necessárias, incluindo:
     - Agrupamentos, sumarizações e cálculos com campos de data.
     - Renomeiação de colunas.
   - **Refined Bucket (S3)**: Os dados processados são armazenados em formato Parquet, particionados por data e pela abreviação da ação do pregão da Bolsa.
   - **Glue Catalog**: Os dados refinados são catalogados automaticamente para consulta no Amazon Athena.
   - **Amazon Athena**: Os dados ficam disponíveis para consulta e análise pelo cliente final.

![Arquitetura](https://github.com/user-attachments/assets/9f71013d-8e1f-45c6-a737-37dd82c9b005)


## Requisitos Atendidos

### Requisitos Obrigatórios

1. **Scrap de Dados**
   - Extração de dados do site da B3 utilizando um script Python.

2. **Ingestão no S3**
   - Os dados brutos são salvos em formato Parquet, particionados por data.

3. **Trigger Lambda**
   - Uma função Lambda é acionada automaticamente ao inserir novos dados no bucket raw.

4. **Iniciação do Job Glue**
   - A Lambda inicia o job de ETL no Glue.

5. **Transformações no Glue (ETL)**
   - Agrupamento e sumarização.
   - Renomeação de colunas.
   - Cálculos com campos de data.

6. **Armazenamento Refinado**
   - Os dados processados são armazenados no bucket refined em formato Parquet.
   - Particionados por data e abreviação do nome da ação.

7. **Catalogação Automática no Glue Catalog**
   - Os dados refinados são automaticamente catalogados no banco de dados padrão do Glue.

8. **Disponibilidade no Athena**
   - Os dados estão legíveis e consultáveis no Amazon Athena.

### Requisitos Opcionais

- **Notebook no Athena**
  - Análise dos dados no Athena utilizando notebooks gráficos.

## Tecnologias Utilizadas

- **AWS Services**:
  - Amazon S3
  - AWS Lambda
  - AWS Glue
  - AWS Glue Catalog
  - Amazon Athena
- **Python**
  - Selenium para web scraping.
  - Bibliotecas para processamento e salva de dados em Parquet.

## Como Executar o Projeto

1. **Configuração Inicial**:
   - Configure sua conta AWS com permissões adequadas para S3, Lambda, Glue e Athena.
   - Instale as dependências necessárias para executar o script de scraping.

2. **Execução do Script de Scrap**:
   - Execute o script Python para baixar os dados da B3 e salvar no bucket raw do S3.

3. **Processamento Automático**:
   - O pipeline será executado automaticamente, acionado pela inserção de novos dados no bucket raw.

4. **Consulta e Análise**:
   - Acesse o Amazon Athena para consultar e analisar os dados processados.

## Estrutura do Repositório

```
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
|   |-- agr_com.sql
|-- IBOVDia/
|   |-- pla.txt
```

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
