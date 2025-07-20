# Dockerfile para Decision ML API
FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro para aproveitar cache do Docker
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY api_main_fixed.py ./api_main.py
COPY best_clustering_model_final.json .
COPY clustering_final_results.json .
COPY dados_processados_final.csv .

# Criar usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expor porta da API
EXPOSE 8000

# Configurar variáveis de ambiente
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
            CMD curl -f http://localhost:8000/health || exit 1

# Comando para iniciar a aplicação
CMD ["python", "-m", "uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]

