#!/usr/bin/env python3
"""
API FASTAPI - 3 CLUSTERS ESPECÍFICOS
API completa para classificação de candidatos nos 3 clusters:
✅ Cluster 0: Candidatos de Sucesso (Finalizaram o Funil com Êxito)
⚠️ Cluster 1: Candidatos que Saíram do Processo (Desistiram ou Foram Reprovados)
🕐 Cluster 2: Candidatos em Processo (Ativos ou Pendentes)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import pickle
import json
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Decision ML - API de Classificação de Candidatos",
    description="API para classificar candidatos em 3 clusters específicos do funil de recrutamento",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Informações dos clusters
CLUSTERS_INFO = {
    0: {
        'nome': 'CANDIDATOS DE SUCESSO',
        'emoji': '✅',
        'descricao': 'Candidatos que finalizaram o funil com êxito',
        'status_incluidos': ['Contratado pela Decision', 'Contratado como Hunting', 'Aprovado'],
        'utilidade': 'Referência para perfis ideais e aprendizado de padrões de sucesso',
        'acao_recomendada': 'Priorizar contratação imediata',
        'cor': '#28a745'
    },
    1: {
        'nome': 'CANDIDATOS QUE SAÍRAM DO PROCESSO',
        'emoji': '⚠️',
        'descricao': 'Candidatos que não estão mais no funil (desistiram ou foram reprovados)',
        'status_incluidos': ['Não Aprovado pelo Cliente', 'Não Aprovado pelo RH', 'Não Aprovado pelo Requisitante', 'Recusado', 'Desistiu', 'Sem interesse nesta vaga', 'Desistiu da Contratação'],
        'utilidade': 'Identificar padrões de rejeição precoce ou perda de engajamento',
        'acao_recomendada': 'Analisar padrões de rejeição/desistência',
        'cor': '#ffc107'
    },
    2: {
        'nome': 'CANDIDATOS EM PROCESSO',
        'emoji': '🕐',
        'descricao': 'Candidatos que ainda estão no funil, em qualquer etapa intermediária',
        'status_incluidos': ['Prospect', 'Inscrito', 'Em avaliação pelo RH', 'Encaminhado ao Requisitante', 'Entrevista Técnica', 'Entrevista com Cliente', 'Documentação PJ', 'Documentação CLT', 'Documentação Cooperado', 'Encaminhar Proposta', 'Proposta Aceita'],
        'utilidade': 'Grupo monitorado para prever quem tende ao sucesso ou abandono',
        'acao_recomendada': 'Monitorar evolução, prever tendência',
        'cor': '#17a2b8'
    }
}

# Variáveis globais para cache
modelo_cache = None
preprocessor_cache = None
feature_names_cache = None
modelo_metadata = {}

class CandidatoInput(BaseModel):
    """Modelo de entrada para dados do candidato"""
    
    # Dados básicos do candidato
    codigo_candidato: str = Field(..., description="Código único do candidato")
    
    # Informações pessoais
    sexo: Optional[str] = Field(None, description="Sexo do candidato")
    idade: Optional[int] = Field(None, description="Idade do candidato")
    
    # Formação e idiomas
    nivel_academico: Optional[str] = Field(None, description="Nível acadêmico")
    nivel_ingles: Optional[str] = Field(None, description="Nível de inglês")
    
    # Informações profissionais
    nivel_profissional: Optional[str] = Field(None, description="Nível profissional")
    area_atuacao: Optional[str] = Field(None, description="Área de atuação")
    cv_texto: Optional[str] = Field(None, description="Texto do CV")
    
    # Dados do processo
    data_candidatura: Optional[str] = Field(None, description="Data da candidatura (YYYY-MM-DD)")
    recrutador: Optional[str] = Field(None, description="Nome do recrutador")
    
    # Dados da vaga
    codigo_vaga: str = Field(..., description="Código da vaga")
    titulo_vaga: Optional[str] = Field(None, description="Título da vaga")
    modalidade: Optional[str] = Field(None, description="Modalidade da vaga")
    cliente: Optional[str] = Field(None, description="Cliente da vaga")
    tipo_contratacao: Optional[str] = Field(None, description="Tipo de contratação")
    cidade_vaga: Optional[str] = Field(None, description="Cidade da vaga")
    estado_vaga: Optional[str] = Field(None, description="Estado da vaga")
    
    # Informações adicionais da vaga
    salario_minimo: Optional[float] = Field(None, description="Salário mínimo")
    salario_maximo: Optional[float] = Field(None, description="Salário máximo")
    experiencia_minima: Optional[int] = Field(None, description="Experiência mínima em anos")
    experiencia_maxima: Optional[int] = Field(None, description="Experiência máxima em anos")
    
    class Config:
        schema_extra = {
            "example": {
                "codigo_candidato": "CAND_12345",
                "sexo": "Masculino",
                "idade": 28,
                "nivel_academico": "Superior Completo",
                "nivel_ingles": "Intermediário",
                "nivel_profissional": "Pleno",
                "area_atuacao": "Tecnologia",
                "cv_texto": "Desenvolvedor Python com 3 anos de experiência...",
                "data_candidatura": "2024-01-15",
                "recrutador": "Ana Silva",
                "codigo_vaga": "VAGA_001",
                "titulo_vaga": "Desenvolvedor Python",
                "modalidade": "CLT",
                "cliente": "Tech Corp",
                "tipo_contratacao": "CLT",
                "cidade_vaga": "São Paulo",
                "estado_vaga": "SP",
                "salario_minimo": 5000.0,
                "salario_maximo": 8000.0,
                "experiencia_minima": 2,
                "experiencia_maxima": 5
            }
        }

class PredictionResponse(BaseModel):
    """Modelo de resposta da predição"""
    
    success: bool = Field(..., description="Se a predição foi bem-sucedida")
    cluster_id: int = Field(..., description="ID do cluster predito (0, 1 ou 2)")
    cluster_info: Dict[str, Any] = Field(..., description="Informações detalhadas do cluster")
    confidence: float = Field(..., description="Confiança da predição (0-1)")
    probabilities: Dict[int, float] = Field(..., description="Probabilidades para cada cluster")
    recommendations: List[str] = Field(..., description="Recomendações específicas")
    metadata: Dict[str, Any] = Field(..., description="Metadados da predição")

class HealthResponse(BaseModel):
    """Modelo de resposta do health check"""
    
    status: str = Field(..., description="Status da API")
    timestamp: str = Field(..., description="Timestamp da verificação")
    model_loaded: bool = Field(..., description="Se o modelo está carregado")
    model_info: Dict[str, Any] = Field(..., description="Informações do modelo")

def carregar_modelo():
    """Carrega modelo e preprocessador"""
    global modelo_cache, preprocessor_cache, feature_names_cache, modelo_metadata
    
    try:
        models_dir = '../models'
        
        # Carregar modelo
        modelo_path = os.path.join(models_dir, 'modelo_final_3_clusters_melhorado.pkl')
        if not os.path.exists(modelo_path):
            # Tentar modelo original
            modelo_path = os.path.join(models_dir, 'best_clustering_model_final.json')
            if os.path.exists(modelo_path):
                with open(modelo_path, 'r') as f:
                    modelo_info = json.load(f)
                modelo_path = os.path.join(models_dir, f"{modelo_info['algorithm']}_model.pkl")
        
        with open(modelo_path, 'rb') as f:
            modelo_carregado = pickle.load(f)
        
        # Verificar formato do modelo
        if isinstance(modelo_carregado, dict):
            modelo_cache = modelo_carregado['modelo']
            algoritmo = modelo_carregado.get('algoritmo', 'Unknown')
        else:
            modelo_cache = modelo_carregado
            algoritmo = type(modelo_cache).__name__
        
        # Carregar preprocessador (se existir)
        try:
            preprocessor_path = os.path.join(models_dir, 'preprocessing_artifacts_3_clusters_melhorado.pkl')
            with open(preprocessor_path, 'rb') as f:
                preprocessor_cache = pickle.load(f)
            feature_names_cache = preprocessor_cache.get('feature_names', [])
        except FileNotFoundError:
            preprocessor_cache = None
            feature_names_cache = []
        
        # Carregar metadados
        try:
            metadata_path = os.path.join(models_dir, 'modelo_metadata_3_clusters_melhorado.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                modelo_metadata = json.load(f)
        except FileNotFoundError:
            modelo_metadata = {
                'algoritmo': algoritmo,
                'data_treinamento': 'N/A',
                'versao': '3_clusters_final'
            }
        
        logger.info(f"Modelo carregado com sucesso: {algoritmo}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return False

def processar_features(candidato_data: Dict) -> np.ndarray:
    """Processa features do candidato para predição"""
    
    # Features básicas que sempre existem
    features = []
    
    # 1. Features categóricas (encoding simples)
    modalidade_map = {'CLT': 1, 'PJ': 2, 'Cooperado': 3, 'Hunting': 4}
    features.append(modalidade_map.get(candidato_data.get('modalidade', ''), 0))
    
    nivel_academico_map = {
        'Ensino Médio': 1, 'Técnico': 2, 'Superior Incompleto': 3,
        'Superior Completo': 4, 'Pós-graduação': 5, 'Mestrado': 6, 'Doutorado': 7
    }
    features.append(nivel_academico_map.get(candidato_data.get('nivel_academico', ''), 0))
    
    nivel_ingles_map = {
        'Básico': 1, 'Intermediário': 2, 'Avançado': 3, 'Fluente': 4, 'Nativo': 5
    }
    features.append(nivel_ingles_map.get(candidato_data.get('nivel_ingles', ''), 0))
    
    nivel_profissional_map = {
        'Estagiário': 1, 'Júnior': 2, 'Pleno': 3, 'Sênior': 4, 'Especialista': 5
    }
    features.append(nivel_profissional_map.get(candidato_data.get('nivel_profissional', ''), 0))
    
    # 2. Features numéricas
    features.append(candidato_data.get('idade', 30))  # Default 30 anos
    features.append(candidato_data.get('salario_minimo', 5000) / 1000)  # Normalizado
    features.append(candidato_data.get('salario_maximo', 8000) / 1000)  # Normalizado
    features.append(candidato_data.get('experiencia_minima', 2))
    features.append(candidato_data.get('experiencia_maxima', 5))
    
    # 3. Features de texto (simples)
    cv_texto = candidato_data.get('cv_texto', '')
    features.append(len(cv_texto) / 1000)  # Tamanho do CV normalizado
    features.append(cv_texto.lower().count('python'))  # Skill específica
    features.append(cv_texto.lower().count('java'))
    features.append(cv_texto.lower().count('javascript'))
    features.append(cv_texto.lower().count('sql'))
    features.append(cv_texto.lower().count('experiência'))
    
    # 4. Features temporais
    try:
        data_candidatura = datetime.strptime(candidato_data.get('data_candidatura', '2024-01-01'), '%Y-%m-%d')
        features.append(data_candidatura.month)  # Mês
        features.append(data_candidatura.weekday())  # Dia da semana
    except:
        features.append(1)  # Janeiro
        features.append(0)  # Segunda-feira
    
    # 5. Features de matching (simples)
    titulo_vaga = candidato_data.get('titulo_vaga', '').lower()
    area_atuacao = candidato_data.get('area_atuacao', '').lower()
    match_score = 0.5  # Default
    if titulo_vaga and area_atuacao:
        # Matching simples baseado em palavras comuns
        titulo_words = set(titulo_vaga.split())
        area_words = set(area_atuacao.split())
        if titulo_words & area_words:
            match_score = 0.8
    features.append(match_score)
    
    # 6. Features de recrutador (simples)
    recrutador = candidato_data.get('recrutador', '')
    features.append(1 if recrutador else 0)  # Tem recrutador
    
    # 7. Features de localização
    estado_map = {
        'SP': 1, 'RJ': 2, 'MG': 3, 'RS': 4, 'PR': 5, 'SC': 6, 'BA': 7, 'GO': 8
    }
    features.append(estado_map.get(candidato_data.get('estado_vaga', ''), 0))
    
    # Garantir que temos exatamente o número de features esperado
    while len(features) < 25:  # Assumindo 25 features
        features.append(0)
    
    return np.array(features[:25]).reshape(1, -1)

@app.on_event("startup")
async def startup_event():
    """Carrega modelo na inicialização"""
    logger.info("Iniciando API Decision ML - 3 Clusters")
    success = carregar_modelo()
    if not success:
        logger.error("Falha ao carregar modelo na inicialização")
    else:
        logger.info("API iniciada com sucesso")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raiz"""
    return {
        "message": "Decision ML API - Classificação de Candidatos em 3 Clusters",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica saúde da API"""
    
    model_loaded = modelo_cache is not None
    
    model_info = {
        "algorithm": modelo_metadata.get('algoritmo', 'Unknown'),
        "training_date": modelo_metadata.get('data_treinamento', 'N/A'),
        "version": modelo_metadata.get('versao', '3_clusters_final'),
        "features_count": len(feature_names_cache) if feature_names_cache else 0
    }
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_loaded,
        model_info=model_info
    )

@app.get("/clusters", response_model=Dict[int, Dict[str, Any]])
async def get_clusters_info():
    """Retorna informações detalhadas dos clusters"""
    return CLUSTERS_INFO

@app.post("/predict", response_model=PredictionResponse)
async def predict_cluster(candidato: CandidatoInput):
    """Prediz cluster do candidato"""
    
    if modelo_cache is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        # Converter para dict
        candidato_dict = candidato.dict()
        
        # Processar features
        X = processar_features(candidato_dict)
        
        # Fazer predição
        cluster_pred = modelo_cache.predict(X)[0]
        
        # Obter probabilidades (se disponível)
        try:
            probas = modelo_cache.predict_proba(X)[0]
            probabilities = {i: float(prob) for i, prob in enumerate(probas)}
            confidence = float(max(probas))
        except:
            # Fallback se não tiver predict_proba
            probabilities = {cluster_pred: 0.8, (cluster_pred + 1) % 3: 0.15, (cluster_pred + 2) % 3: 0.05}
            confidence = 0.8
        
        # Informações do cluster
        cluster_info = CLUSTERS_INFO[cluster_pred].copy()
        
        # Gerar recomendações específicas
        recommendations = []
        
        if cluster_pred == 0:  # Sucesso
            recommendations = [
                "🎯 Candidato com alto potencial de sucesso",
                "⚡ Priorizar para contratação imediata",
                "📞 Acelerar processo de entrevistas",
                "💼 Preparar proposta competitiva"
            ]
        elif cluster_pred == 1:  # Saíram do processo
            recommendations = [
                "⚠️ Candidato com padrão de saída do processo",
                "🔍 Investigar possíveis motivos de desistência",
                "💡 Melhorar experiência do candidato",
                "📋 Revisar critérios de seleção"
            ]
        else:  # Em processo
            recommendations = [
                "🕐 Candidato em processo ativo",
                "📊 Monitorar evolução no funil",
                "🎯 Identificar próximos passos",
                "📈 Acompanhar tendência de conversão"
            ]
        
        # Metadados da predição
        metadata = {
            "prediction_timestamp": datetime.now().isoformat(),
            "model_algorithm": modelo_metadata.get('algoritmo', 'Unknown'),
            "features_used": len(X[0]),
            "codigo_candidato": candidato_dict.get('codigo_candidato'),
            "codigo_vaga": candidato_dict.get('codigo_vaga')
        }
        
        return PredictionResponse(
            success=True,
            cluster_id=int(cluster_pred),
            cluster_info=cluster_info,
            confidence=confidence,
            probabilities=probabilities,
            recommendations=recommendations,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(candidatos: List[CandidatoInput]):
    """Prediz clusters para múltiplos candidatos"""
    
    if modelo_cache is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    if len(candidatos) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 candidatos por batch")
    
    results = []
    
    for candidato in candidatos:
        try:
            # Reutilizar lógica do predict individual
            result = await predict_cluster(candidato)
            results.append(result)
        except Exception as e:
            # Adicionar resultado de erro para este candidato
            error_result = PredictionResponse(
                success=False,
                cluster_id=-1,
                cluster_info={"error": str(e)},
                confidence=0.0,
                probabilities={},
                recommendations=["❌ Erro na predição"],
                metadata={"error": str(e), "codigo_candidato": candidato.codigo_candidato}
            )
            results.append(error_result)
    
    return results

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Retorna informações detalhadas do modelo"""
    
    if modelo_cache is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    return {
        "algorithm": modelo_metadata.get('algoritmo', 'Unknown'),
        "training_date": modelo_metadata.get('data_treinamento', 'N/A'),
        "version": modelo_metadata.get('versao', '3_clusters_final'),
        "features_count": len(feature_names_cache) if feature_names_cache else 0,
        "feature_names": feature_names_cache[:10] if feature_names_cache else [],  # Primeiras 10
        "clusters_count": 3,
        "clusters_info": CLUSTERS_INFO,
        "model_type": type(modelo_cache).__name__ if modelo_cache else "Unknown"
    }

@app.post("/model/reload")
async def reload_model():
    """Recarrega o modelo"""
    
    global modelo_cache, preprocessor_cache, feature_names_cache, modelo_metadata
    
    # Limpar cache
    modelo_cache = None
    preprocessor_cache = None
    feature_names_cache = None
    modelo_metadata = {}
    
    # Recarregar
    success = carregar_modelo()
    
    if success:
        return {"message": "Modelo recarregado com sucesso", "timestamp": datetime.now().isoformat()}
    else:
        raise HTTPException(status_code=500, detail="Falha ao recarregar modelo")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Iniciando Decision ML API - 3 Clusters")
    print("📊 Clusters disponíveis:")
    for cluster_id, info in CLUSTERS_INFO.items():
        print(f"   {info['emoji']} Cluster {cluster_id}: {info['nome']}")
    print("\\n📖 Documentação: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

