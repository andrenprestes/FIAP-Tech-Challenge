#!/bin/bash

# Script de Deploy - Decision ML
# ==============================

set -e  # Parar em caso de erro

echo "🚀 INICIANDO DEPLOY DA DECISION ML API"
echo "======================================"

# Cores para output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# Função para log colorido
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verificar se Docker está instalado
if ! command -v docker &> /dev/null; then
    log_error "Docker não está instalado!"
    exit 1
fi

# Verificar se Docker Compose está instalado
if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose não está instalado!"
    exit 1
fi

# Criar diretório de logs
log_info "Criando diretório de logs..."
mkdir -p logs
log_success "Diretório de logs criado"

# Parar containers existentes
log_info "Parando containers existentes..."
docker-compose down --remove-orphans || true
log_success "Containers parados"

# Limpar imagens antigas (opcional)
if [ "$1" = "--clean" ]; then
    log_info "Limpando imagens antigas..."
    docker system prune -f
    log_success "Limpeza concluída"
fi

# Build da aplicação
log_info "Construindo imagem da aplicação..."
docker-compose build decision-ml-api
log_success "Imagem construída com sucesso"

# Iniciar serviços
log_info "Iniciando serviços..."
docker-compose up -d

# Aguardar inicialização
log_info "Aguardando inicialização dos serviços..."
sleep 30

# Verificar status dos containers
log_info "Verificando status dos containers..."
docker-compose ps

# Testar API
log_info "Testando API..."
max_attempts=10
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API está respondendo!"
        break
    else
        log_warning "Tentativa $attempt/$max_attempts - API ainda não está pronta"
        sleep 5
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    log_error "API não respondeu após $max_attempts tentativas"
    log_info "Verificando logs da API..."
    docker-compose logs decision-ml-api
    exit 1
fi

# Executar testes básicos
log_info "Executando testes básicos..."

# Teste 1: Health check
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    log_success "Health check passou"
else
    log_error "Health check falhou"
fi

# Teste 2: Model info
if curl -s http://localhost:8000/model-info | grep -q "DBSCAN"; then
    log_success "Model info passou"
else
    log_error "Model info falhou"
fi

# Teste 3: Predição simples
test_data='{
    "ano_candidatura": 2024,
    "mes_candidatura": 7,
    "dia_semana": 2,
    "trimestre": 3,
    "taxa_sucesso_recrutador": 0.35,
    "volume_recrutador": 50,
    "taxa_sucesso_modalidade": 0.08,
    "posicao_funil": 2,
    "modalidade_encoded": 1,
    "recrutador_clean_encoded": 5
}'

if curl -s -X POST -H "Content-Type: application/json" -d "$test_data" http://localhost:8000/predict | grep -q "cluster_id"; then
    log_success "Teste de predição passou"
else
    log_error "Teste de predição falhou"
fi

# Mostrar informações de acesso
echo ""
echo "======================================"
log_success "DEPLOY CONCLUÍDO COM SUCESSO!"
echo "======================================"
echo ""
echo "🌐 Serviços disponíveis:"
echo "   • API Principal: http://localhost:8000"
echo "   • Documentação: http://localhost:8000/docs"
echo "   • Health Check: http://localhost:8000/health"
echo "   • MLflow UI: http://localhost:5000"
echo "   • Nginx (Proxy): http://localhost:80"
echo "   • Prometheus: http://localhost:9090"
echo "   • Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "📋 Comandos úteis:"
echo "   • Ver logs: docker-compose logs -f [service]"
echo "   • Parar: docker-compose down"
echo "   • Reiniciar: docker-compose restart [service]"
echo "   • Status: docker-compose ps"
echo ""
echo "🔧 Para monitoramento:"
echo "   • Logs da API: docker-compose logs -f decision-ml-api"
echo "   • Métricas: http://localhost:9090"
echo "   • Dashboards: http://localhost:3000"
echo ""

# Mostrar logs recentes da API
log_info "Logs recentes da API:"
docker-compose logs --tail=20 decision-ml-api

echo ""
log_success "Deploy finalizado! 🎉"

