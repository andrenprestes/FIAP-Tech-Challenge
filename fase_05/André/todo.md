# TODO - Projeto Decision ML

## Fase 1: Análise Exploratória dos Dados
- [x] Carregar e examinar estrutura dos dados (applicants.json, vagas.json, prospects.json)
- [x] Análise de dimensões e tipos de dados
- [x] Identificação de valores ausentes e padrões
- [x] Análise de duplicatas
- [ ] Detecção de outliers
- [x] Distribuições de variáveis categóricas e numéricas
- [ ] Análise de correlações
- [x] Visualizações exploratórias
- [x] Análise temporal se aplicável
- [x] Documentar insights iniciais

## Fase 2: Processamento e Limpeza dos Dados
- [x] Implementar pipeline de limpeza
- [x] Tratamento de duplicatas
- [x] Tratamento de valores ausentes (máx 30% missing)
- [x] Detecção e tratamento de outliers
- [x] Validação de consistência e ranges válidos
- [x] Padronização de textos e categorias
- [x] Feature engineering
- [x] Encoding categórico
- [x] Normalização e padronização
- [ ] Balanceamento de dados (será feito na fase de clustering)

## Fase 3: Treinamento de Modelos de Clustering
- [x] Configurar MLflow tracking
- [x] Implementar K-means
- [x] Implementar DBSCAN
- [x] Implementar clustering hierárquico
- [x] Otimização de hiperparâmetros
- [x] Validação cruzada
- [x] Comparação de algoritmos
- [x] Seleção do modelo final (DBSCAN - Silhouette: 0.645)
- [x] Versionamento de modelos

## Fase 4: Avaliação e Interpretação dos Modelos
- [x] Calcular métricas de clustering (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- [x] Análise e interpretação dos clusters
- [x] Visualizações dos perfis identificados
- [x] Insights de negócio
- [x] Validação da qualidade dos clusters
- [x] Relatório de avaliação

## Fase 5: Desenvolvimento da API
- [x] Estrutura modular da API FastAPI
- [x] Endpoint /predict
- [x] Endpoint /health
- [x] Endpoint /model-info
- [x] Validação com Pydantic
- [x] Tratamento de erros
- [x] Documentação OpenAPI
- [x] Logging estruturado
- [x] Testes unitários

## Fase 6: Containerização e Deploy
- [x] Dockerfile otimizado
- [x] Docker-compose com MLflow UI
- [x] Configuração via variáveis de ambiente
- [x] Volumes para persistência
- [x] Rede entre serviços
- [x] Testes de integração
- [x] Sistema de monitoramento
- [x] Métricas de performance

## Fase 7: Documentação e Entrega
- [x] README detalhado
- [x] Guia de uso dos notebooks
- [x] Documentação da API
- [x] Relatório técnico
- [x] Instruções de deploy
- [x] Exemplos práticos
- [x] Entrega final

