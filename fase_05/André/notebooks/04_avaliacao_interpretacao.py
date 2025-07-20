#!/usr/bin/env python3
"""
AVALIAÇÃO DO MODELO - 3 CLUSTERS ESPECÍFICOS
Avaliação completa do modelo treinado para os 3 clusters:
✅ Cluster 0: Candidatos de Sucesso (Finalizaram o Funil com Êxito)
⚠️ Cluster 1: Candidatos que Saíram do Processo (Desistiram ou Foram Reprovados)
🕐 Cluster 2: Candidatos em Processo (Ativos ou Pendentes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class AvaliadorModelo3Clusters:
    def __init__(self):
        self.models_dir = '../models'
        self.reports_dir = '../reports'
        self.visualizations_dir = '../visualizations'
        
        # Criar diretórios
        for directory in [self.reports_dir, self.visualizations_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Informações dos clusters
        self.clusters_info = {
            0: {'nome': 'CANDIDATOS DE SUCESSO', 'emoji': '✅', 'cor': '#28a745'},
            1: {'nome': 'CANDIDATOS QUE SAÍRAM DO PROCESSO', 'emoji': '⚠️', 'cor': '#ffc107'},
            2: {'nome': 'CANDIDATOS EM PROCESSO', 'emoji': '🕐', 'cor': '#17a2b8'}
        }
        
        # Critérios de qualidade específicos por cluster
        self.criterios_qualidade = {
            0: {  # Cluster Sucesso - CRÍTICO
                'precision_min': 0.70,  # 70% - Não perder talentos
                'recall_min': 0.60,     # 60% - Identificar a maioria
                'f1_min': 0.65,         # 65% - Balanceado
                'importancia': 'CRÍTICA'
            },
            1: {  # Cluster Saíram - ALTA
                'precision_min': 0.75,  # 75% - Identificar padrões de saída
                'recall_min': 0.70,     # 70% - Capturar a maioria
                'f1_min': 0.72,         # 72% - Balanceado
                'importancia': 'ALTA'
            },
            2: {  # Cluster Processo - MÉDIA
                'precision_min': 0.80,  # 80% - Maioria está em processo
                'recall_min': 0.85,     # 85% - Capturar bem
                'f1_min': 0.82,         # 82% - Balanceado
                'importancia': 'MÉDIA'
            }
        }
        
        # Cores dos clusters
        self.cores_clusters = ['#28a745', '#ffc107', '#17a2b8']
        
        # Resultados da avaliação
        self.resultados = {}
    
    def carregar_modelo_e_dados(self):
        """Carrega modelo treinado e dados de teste"""
        print("🔄 CARREGANDO MODELO E DADOS DE TESTE...")
        
        try:
            # Carregar modelo
            modelo_path = os.path.join(self.models_dir, 'modelo_final_3_clusters_melhorado.pkl')
            with open(modelo_path, 'rb') as f:
                modelo_carregado = pickle.load(f)
            
            # Verificar se é um dicionário ou modelo direto
            if isinstance(modelo_carregado, dict):
                # Formato esperado: dicionário com informações
                modelo = modelo_carregado['modelo']
                algoritmo = modelo_carregado.get('algoritmo', 'Unknown')
                score_cv = modelo_carregado.get('score_cv', 0)
                score_test = modelo_carregado.get('score_test', 0)
            else:
                # Formato direto: apenas o modelo
                modelo = modelo_carregado
                algoritmo = type(modelo).__name__
                score_cv = 0
                score_test = 0
                print("⚠️ Modelo carregado em formato direto (sem metadados)")
            
            print(f"✓ Modelo carregado: {algoritmo}")
            print(f"✓ Score CV: {score_cv:.3f}")
            print(f"✓ Score Test: {score_test:.3f}")
            
            # Carregar dados processados
            dados_path = os.path.join(self.models_dir, 'processed_data_3_clusters_melhorado.pkl')
            with open(dados_path, 'rb') as f:
                dados_carregados = pickle.load(f)
            
            # Verificar formato dos dados
            if isinstance(dados_carregados, dict):
                # Formato esperado: dicionário com dados
                X_test = dados_carregados['X_test']
                y_test = dados_carregados['y_test']
                feature_names = dados_carregados.get('feature_names', [f'feature_{i}' for i in range(X_test.shape[1])])
            else:
                # Formato alternativo: assumir que são os dados diretamente
                print("⚠️ Formato de dados não reconhecido, tentando carregar dados originais...")
                # Tentar carregar dados do processamento original
                try:
                    with open('../models/processed_data_3_clusters_melhorado.pkl', 'rb') as f:
                        dados_originais = pickle.load(f)
                    X_test = dados_originais['X_test']
                    y_test = dados_originais['y_test']
                    feature_names = dados_originais.get('feature_names', [f'feature_{i}' for i in range(X_test.shape[1])])
                except:
                    raise FileNotFoundError("Não foi possível carregar dados de teste válidos")
            
            print(f"✓ Dados de teste: {len(X_test)} registros")
            print(f"✓ Features: {len(feature_names)}")
            
            # Carregar metadados (opcional)
            metadata = {}
            try:
                metadata_path = os.path.join(self.models_dir, 'modelo_metadata_3_clusters_melhorado.json')
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"✓ Data de treinamento: {metadata.get('data_treinamento', 'N/A')}")
            except FileNotFoundError:
                print("⚠️ Metadados não encontrados, continuando sem eles...")
                metadata = {'data_treinamento': 'N/A'}
            
            return modelo, X_test, y_test, feature_names, metadata, algoritmo
            
        except FileNotFoundError as e:
            print(f"❌ Erro: Arquivo não encontrado - {e}")
            print("💡 Execute primeiro o treinamento: python 03_treinamento_3_CLUSTERS_FINAL.py")
            raise
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise

    
    def calcular_metricas_basicas(self, y_true, y_pred):
        """Calcula métricas básicas de classificação"""
        print("\\n=== CALCULANDO MÉTRICAS BÁSICAS ===")
        
        # Métricas gerais
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"📊 Accuracy: {accuracy:.3f} ({accuracy:.1%})")
        print(f"📊 Precision (Macro): {precision_macro:.3f}")
        print(f"📊 Recall (Macro): {recall_macro:.3f}")
        print(f"📊 F1-Score (Macro): {f1_macro:.3f}")
        print(f"📊 F1-Score (Weighted): {f1_weighted:.3f}")
        
        # Métricas por cluster
        precision_por_cluster = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_por_cluster = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_por_cluster = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        print(f"\\n📊 MÉTRICAS POR CLUSTER:")
        metricas_clusters = {}
        
        for cluster_id in range(3):
            if cluster_id < len(precision_por_cluster):
                info = self.clusters_info[cluster_id]
                precision = precision_por_cluster[cluster_id]
                recall = recall_por_cluster[cluster_id]
                f1 = f1_por_cluster[cluster_id]
                
                print(f"   {info['emoji']} Cluster {cluster_id} - {info['nome']}:")
                print(f"      Precision: {precision:.3f} ({precision:.1%})")
                print(f"      Recall: {recall:.3f} ({recall:.1%})")
                print(f"      F1-Score: {f1:.3f} ({f1:.1%})")
                
                metricas_clusters[cluster_id] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                }
        
        # Salvar resultados
        self.resultados['metricas_basicas'] = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'metricas_por_cluster': metricas_clusters
        }
        
        return accuracy, f1_weighted, metricas_clusters
    
    def avaliar_qualidade_por_cluster(self, metricas_clusters):
        """Avalia qualidade específica por cluster"""
        print("\\n=== AVALIAÇÃO DE QUALIDADE POR CLUSTER ===")
        
        clusters_aprovados = 0
        avaliacoes = {}
        
        for cluster_id, metricas in metricas_clusters.items():
            info = self.clusters_info[cluster_id]
            criterios = self.criterios_qualidade[cluster_id]
            
            precision = metricas['precision']
            recall = metricas['recall']
            f1 = metricas['f1_score']
            
            # Verificar critérios
            precision_ok = precision >= criterios['precision_min']
            recall_ok = recall >= criterios['recall_min']
            f1_ok = f1 >= criterios['f1_min']
            
            aprovado = precision_ok and recall_ok and f1_ok
            if aprovado:
                clusters_aprovados += 1
            
            # Status
            if aprovado:
                status = "✅ APROVADO"
                status_emoji = "✅"
            elif precision_ok and f1_ok:
                status = "⚠️ PARCIAL"
                status_emoji = "⚠️"
            else:
                status = "❌ REPROVADO"
                status_emoji = "❌"
            
            print(f"\\n{info['emoji']} CLUSTER {cluster_id} - {info['nome']} ({criterios['importancia']}):")
            print(f"   📊 Precision: {precision:.3f} (mín: {criterios['precision_min']:.2f}) {'✅' if precision_ok else '❌'}")
            print(f"   📊 Recall: {recall:.3f} (mín: {criterios['recall_min']:.2f}) {'✅' if recall_ok else '❌'}")
            print(f"   📊 F1-Score: {f1:.3f} (mín: {criterios['f1_min']:.2f}) {'✅' if f1_ok else '❌'}")
            print(f"   🎯 Status: {status}")
            
            avaliacoes[cluster_id] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'precision_ok': precision_ok,
                'recall_ok': recall_ok,
                'f1_ok': f1_ok,
                'aprovado': aprovado,
                'status': status,
                'status_emoji': status_emoji,
                'importancia': criterios['importancia']
            }
        
        # Salvar resultados
        self.resultados['avaliacao_qualidade'] = {
            'clusters_aprovados': clusters_aprovados,
            'total_clusters': 3,
            'taxa_aprovacao': clusters_aprovados / 3,
            'avaliacoes_por_cluster': avaliacoes
        }
        
        return clusters_aprovados, avaliacoes
    
    def calcular_score_negocio(self, accuracy, f1_weighted, clusters_aprovados, avaliacoes):
        """Calcula score de negócio personalizado"""
        print("\\n=== CALCULANDO SCORE DE NEGÓCIO ===")
        
        # Pesos por importância
        pesos_importancia = {
            'CRÍTICA': 0.5,   # Cluster 0 (Sucesso) - 50%
            'ALTA': 0.3,      # Cluster 1 (Saíram) - 30%
            'MÉDIA': 0.2      # Cluster 2 (Processo) - 20%
        }
        
        # Score ponderado por cluster
        score_clusters = 0
        for cluster_id, avaliacao in avaliacoes.items():
            importancia = avaliacao['importancia']
            peso = pesos_importancia[importancia]
            
            # Score do cluster (média das métricas)
            score_cluster = (avaliacao['precision'] + avaliacao['recall'] + avaliacao['f1_score']) / 3
            score_clusters += score_cluster * peso
        
        # Score geral (30% métricas gerais + 70% clusters ponderados)
        score_geral = (accuracy + f1_weighted) / 2
        score_final = (score_geral * 0.3) + (score_clusters * 0.7)
        
        # Penalidade por clusters reprovados
        penalidade = 0
        for cluster_id, avaliacao in avaliacoes.items():
            if not avaliacao['aprovado']:
                importancia = avaliacao['importancia']
                if importancia == 'CRÍTICA':
                    penalidade += 0.15  # -15% por cluster crítico reprovado
                elif importancia == 'ALTA':
                    penalidade += 0.10  # -10% por cluster alta reprovado
                else:
                    penalidade += 0.05  # -5% por cluster média reprovado
        
        score_final_penalizado = max(0, score_final - penalidade)
        
        # Classificação
        if score_final_penalizado >= 0.85:
            classificacao = "🏆 EXCELENTE"
            recomendacao = "APROVADO PARA PRODUÇÃO"
        elif score_final_penalizado >= 0.75:
            classificacao = "✅ BOM"
            recomendacao = "APROVADO COM MONITORAMENTO"
        elif score_final_penalizado >= 0.65:
            classificacao = "⚠️ ACEITÁVEL"
            recomendacao = "APROVADO COM RESSALVAS"
        elif score_final_penalizado >= 0.50:
            classificacao = "🔧 PRECISA MELHORAR"
            recomendacao = "MELHORIAS NECESSÁRIAS"
        else:
            classificacao = "❌ INADEQUADO"
            recomendacao = "RETREINAR MODELO"
        
        print(f"📊 Score de Clusters (Ponderado): {score_clusters:.3f}")
        print(f"📊 Score Geral: {score_geral:.3f}")
        print(f"📊 Penalidade: -{penalidade:.3f}")
        print(f"📊 Score Final: {score_final_penalizado:.3f} ({score_final_penalizado:.1%})")
        print(f"🎯 Classificação: {classificacao}")
        print(f"🎯 Recomendação: {recomendacao}")
        
        # Salvar resultados
        self.resultados['score_negocio'] = {
            'score_clusters': float(score_clusters),
            'score_geral': float(score_geral),
            'penalidade': float(penalidade),
            'score_final': float(score_final_penalizado),
            'classificacao': classificacao,
            'recomendacao': recomendacao
        }
        
        return score_final_penalizado, classificacao, recomendacao
    
    def gerar_matriz_confusao(self, y_true, y_pred):
        """Gera e visualiza matriz de confusão"""
        print("\\n=== GERANDO MATRIZ DE CONFUSÃO ===")
        
        # Calcular matriz
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalizar por linha (recall)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Visualização
        plt.figure(figsize=(10, 8))
        
        # Matriz absoluta
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'Pred {i}' for i in range(3)],
                   yticklabels=[f'Real {i}' for i in range(3)])
        plt.title('Matriz de Confusão\\n(Valores Absolutos)', fontsize=12, fontweight='bold')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Predita')
        
        # Matriz normalizada
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=[f'Pred {i}' for i in range(3)],
                   yticklabels=[f'Real {i}' for i in range(3)])
        plt.title('Matriz de Confusão\\n(Normalizada por Linha)', fontsize=12, fontweight='bold')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Predita')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'matriz_confusao_3_clusters.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Matriz de confusão salva: matriz_confusao_3_clusters.png")
        
        # Salvar dados da matriz
        self.resultados['matriz_confusao'] = {
            'matriz_absoluta': cm.tolist(),
            'matriz_normalizada': cm_normalized.tolist()
        }
    
    def analisar_distribuicao_predicoes(self, y_true, y_pred):
        """Analisa distribuição das predições vs realidade"""
        print("\\n=== ANÁLISE DE DISTRIBUIÇÃO DAS PREDIÇÕES ===")
        
        # Distribuições
        dist_real = pd.Series(y_true).value_counts().sort_index()
        dist_pred = pd.Series(y_pred).value_counts().sort_index()
        
        print("\\n📊 DISTRIBUIÇÃO REAL vs PREDITA:")
        print("Cluster | Real | Predito | Diferença")
        print("-" * 40)
        
        distribuicoes = {}
        for cluster_id in range(3):
            real = dist_real.get(cluster_id, 0)
            pred = dist_pred.get(cluster_id, 0)
            diff = pred - real
            diff_pct = (diff / real * 100) if real > 0 else 0
            
            info = self.clusters_info[cluster_id]
            print(f"{cluster_id} - {info['emoji']:2} | {real:4d} | {pred:7d} | {diff:+4d} ({diff_pct:+5.1f}%)")
            
            distribuicoes[cluster_id] = {
                'real': int(real),
                'predito': int(pred),
                'diferenca': int(diff),
                'diferenca_pct': float(diff_pct)
            }
        
        # Visualização
        plt.figure(figsize=(12, 6))
        
        x = range(3)
        width = 0.35
        
        plt.bar([i - width/2 for i in x], [dist_real.get(i, 0) for i in range(3)], 
               width, label='Real', color='lightblue', alpha=0.8)
        plt.bar([i + width/2 for i in x], [dist_pred.get(i, 0) for i in range(3)], 
               width, label='Predito', color='orange', alpha=0.8)
        
        plt.title('Distribuição Real vs Predita por Cluster', fontsize=14, fontweight='bold')
        plt.xlabel('Clusters')
        plt.ylabel('Número de Candidatos')
        plt.legend()
        
        # Labels dos clusters
        labels = [f"{self.clusters_info[i]['emoji']} Cluster {i}" for i in range(3)]
        plt.xticks(x, labels)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'distribuicao_predicoes_3_clusters.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Distribuição salva: distribuicao_predicoes_3_clusters.png")
        
        # Salvar resultados
        self.resultados['distribuicao_predicoes'] = distribuicoes
    
    def gerar_relatorio_classificacao(self, y_true, y_pred):
        """Gera relatório detalhado de classificação"""
        print("\\n=== RELATÓRIO DETALHADO DE CLASSIFICAÇÃO ===")
        
        # Relatório sklearn
        target_names = [f"{self.clusters_info[i]['emoji']} Cluster {i}" for i in range(3)]
        report = classification_report(y_true, y_pred, target_names=target_names, 
                                     output_dict=True, zero_division=0)
        
        print("\\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        
        # Salvar relatório
        self.resultados['relatorio_classificacao'] = report
    
    def gerar_insights_melhorias(self, avaliacoes, score_final):
        """Gera insights e sugestões de melhorias"""
        print("\\n=== INSIGHTS E SUGESTÕES DE MELHORIAS ===")
        
        insights = []
        
        # Análise por cluster
        for cluster_id, avaliacao in avaliacoes.items():
            info = self.clusters_info[cluster_id]
            
            if not avaliacao['aprovado']:
                if avaliacao['precision'] < 0.7:
                    insights.append({
                        'tipo': 'precision_baixa',
                        'cluster': cluster_id,
                        'titulo': f'Precision Baixa - {info["nome"]}',
                        'descricao': f'Precision de {avaliacao["precision"]:.1%} no cluster {cluster_id}.',
                        'sugestao': 'Revisar features distintivas e balanceamento de classes.',
                        'prioridade': avaliacao['importancia']
                    })
                
                if avaliacao['recall'] < 0.6:
                    insights.append({
                        'tipo': 'recall_baixo',
                        'cluster': cluster_id,
                        'titulo': f'Recall Baixo - {info["nome"]}',
                        'descricao': f'Recall de {avaliacao["recall"]:.1%} no cluster {cluster_id}.',
                        'sugestao': 'Aumentar dados de treinamento ou ajustar threshold.',
                        'prioridade': avaliacao['importancia']
                    })
        
        # Análise geral
        if score_final < 0.75:
            insights.append({
                'tipo': 'score_geral_baixo',
                'cluster': None,
                'titulo': 'Score Geral Baixo',
                'descricao': f'Score final de {score_final:.1%} abaixo do ideal.',
                'sugestao': 'Considerar ensemble de modelos ou feature engineering avançado.',
                'prioridade': 'CRÍTICA'
            })
        
        # Salvar insights
        self.resultados['insights_melhorias'] = insights
        
        print("\\n🎯 INSIGHTS IDENTIFICADOS:")
        for i, insight in enumerate(insights, 1):
            prioridade_emoji = {'CRÍTICA': '🔴', 'ALTA': '🟡', 'MÉDIA': '🟢'}.get(insight['prioridade'], '⚪')
            print(f"\\n{i}. {prioridade_emoji} {insight['titulo']} ({insight['prioridade']})")
            print(f"   📊 {insight['descricao']}")
            print(f"   💡 {insight['sugestao']}")
    
    def salvar_relatorio_final(self, algoritmo, metadata):
        """Salva relatório final completo"""
        print("\\n💾 SALVANDO RELATÓRIO FINAL...")
        
        # Adicionar metadados
        self.resultados['metadata'] = {
            'data_avaliacao': datetime.now().isoformat(),
            'algoritmo_usado': algoritmo,
            'versao': '3_clusters_final',
            'data_treinamento': metadata.get('data_treinamento'),
            'clusters_definicao': self.clusters_info
        }
        
        # Salvar JSON
        relatorio_path = os.path.join(self.reports_dir, 'avaliacao_modelo_3_clusters.json')
        with open(relatorio_path, 'w', encoding='utf-8') as f:
            json.dump(self.resultados, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ Relatório salvo: {relatorio_path}")
        
        return relatorio_path
    
    def executar_avaliacao_completa(self):
        """Executa avaliação completa do modelo"""
        print("🎯 INICIANDO AVALIAÇÃO COMPLETA DO MODELO - 3 CLUSTERS")
        print("=" * 70)
        
        # 1. Carregar modelo e dados
        modelo, X_test, y_test, feature_names, metadata, algoritmo = self.carregar_modelo_e_dados()
        
        # 2. Fazer predições
        print("\\n🔄 REALIZANDO PREDIÇÕES...")
        y_pred = modelo.predict(X_test)
        print(f"✓ Predições realizadas para {len(y_test)} registros")
        
        # 3. Calcular métricas básicas
        accuracy, f1_weighted, metricas_clusters = self.calcular_metricas_basicas(y_test, y_pred)
        
        # 4. Avaliar qualidade por cluster
        clusters_aprovados, avaliacoes = self.avaliar_qualidade_por_cluster(metricas_clusters)
        
        # 5. Calcular score de negócio
        score_final, classificacao, recomendacao = self.calcular_score_negocio(
            accuracy, f1_weighted, clusters_aprovados, avaliacoes)
        
        # 6. Gerar matriz de confusão
        self.gerar_matriz_confusao(y_test, y_pred)
        
        # 7. Analisar distribuição das predições
        self.analisar_distribuicao_predicoes(y_test, y_pred)
        
        # 8. Gerar relatório de classificação
        self.gerar_relatorio_classificacao(y_test, y_pred)
        
        # 9. Gerar insights e melhorias
        self.gerar_insights_melhorias(avaliacoes, score_final)
        
        # 10. Salvar relatório final
        relatorio_path = self.salvar_relatorio_final(algoritmo, metadata)
        
        # Resumo final
        print("\\n🏆 RELATÓRIO FINAL DE AVALIAÇÃO - 3 CLUSTERS")
        print("=" * 70)
        print(f"🤖 Algoritmo: {algoritmo}")
        print(f"📊 Accuracy: {accuracy:.3f} ({accuracy:.1%})")
        print(f"📊 F1-Score Weighted: {f1_weighted:.3f} ({f1_weighted:.1%})")
        print(f"📊 Score Final: {score_final:.1f}/100 ({score_final:.1%})")
        print(f"🎯 Classificação: {classificacao}")
        print(f"🎯 Recomendação: {recomendacao}")
        
        print(f"\\n📊 PERFORMANCE POR CLUSTER:")
        for cluster_id, avaliacao in avaliacoes.items():
            info = self.clusters_info[cluster_id]
            print(f"   {info['emoji']} Cluster {cluster_id} - {info['nome']}: {avaliacao['status']}")
        
        print(f"\\n📁 Arquivos gerados:")
        print(f"   📊 Visualizações: {self.visualizations_dir}")
        print(f"   📄 Relatório: {relatorio_path}")
        
        return self.resultados

def main():
    """Função principal"""
    avaliador = AvaliadorModelo3Clusters()
    resultados = avaliador.executar_avaliacao_completa()
    
    score_final = resultados['score_negocio']['score_final']
    recomendacao = resultados['score_negocio']['recomendacao']
    
    print(f"\\n✅ Avaliação concluída!")
    print(f"   - Score Final: {score_final:.1%}")
    print(f"   - Recomendação: {recomendacao}")

if __name__ == "__main__":
    main()

