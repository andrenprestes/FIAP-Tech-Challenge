#!/usr/bin/env python3
"""
ANÁLISE EXPLORATÓRIA - 3 CLUSTERS ESPECÍFICOS
Análise focada nos 3 clusters definidos:
✅ Cluster 0: Candidatos de Sucesso (Finalizaram o Funil com Êxito)
⚠️ Cluster 1: Candidatos que Saíram do Processo (Desistiram ou Foram Reprovados)
🕐 Cluster 2: Candidatos em Processo (Ativos ou Pendentes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para português
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class AnalisadorExploratorio3Clusters:
    def __init__(self):
        self.data_dir = '../data'
        self.reports_dir = '../reports'
        self.visualizations_dir = '../visualizations'
        
        # Criar diretórios
        for directory in [self.reports_dir, self.visualizations_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Mapeamento dos 3 clusters
        self.clusters_mapping = {
            # ✅ CLUSTER 0: CANDIDATOS DE SUCESSO
            'Contratado pela Decision': 0,
            'Contratado como Hunting': 0,
            'Aprovado': 0,
            
            # ⚠️ CLUSTER 1: CANDIDATOS QUE SAÍRAM DO PROCESSO
            'Não Aprovado pelo Cliente': 1,
            'Não Aprovado pelo RH': 1,
            'Não Aprovado pelo Requisitante': 1,
            'Recusado': 1,
            'Desistiu': 1,
            'Sem interesse nesta vaga': 1,
            'Desistiu da Contratação': 1,
            
            # 🕐 CLUSTER 2: CANDIDATOS EM PROCESSO
            'Prospect': 2,
            'Inscrito': 2,
            'Em avaliação pelo RH': 2,
            'Encaminhado ao Requisitante': 2,
            'Entrevista Técnica': 2,
            'Entrevista com Cliente': 2,
            'Documentação PJ': 2,
            'Documentação CLT': 2,
            'Documentação Cooperado': 2,
            'Encaminhar Proposta': 2,
            'Proposta Aceita': 2
        }
        
        # Nomes e cores dos clusters
        self.clusters_info = {
            0: {'nome': 'CANDIDATOS DE SUCESSO', 'emoji': '✅', 'cor': '#28a745'},
            1: {'nome': 'CANDIDATOS QUE SAÍRAM DO PROCESSO', 'emoji': '⚠️', 'cor': '#ffc107'},
            2: {'nome': 'CANDIDATOS EM PROCESSO', 'emoji': '🕐', 'cor': '#17a2b8'}
        }
        
        # Paleta de cores
        self.cores_clusters = ['#28a745', '#ffc107', '#17a2b8']
        
        # Insights coletados
        self.insights = {}
    
    def carregar_dados(self):
        """Carrega e prepara dados básicos"""
        print("🔄 CARREGANDO DADOS PARA ANÁLISE EXPLORATÓRIA...")
        
        # Carregar dados
        with open(os.path.join(self.data_dir, 'prospects.json'), 'r', encoding='utf-8') as f:
            prospects_data = json.load(f)
        
        with open(os.path.join(self.data_dir, 'vagas.json'), 'r', encoding='utf-8') as f:
            vagas_data = json.load(f)
        
        with open(os.path.join(self.data_dir, 'applicants.json'), 'r', encoding='utf-8') as f:
            applicants_data = json.load(f)
        
        print(f"✓ Prospects: {len(prospects_data)} vagas")
        print(f"✓ Vagas: {len(vagas_data)} registros")
        print(f"✓ Applicants: {len(applicants_data)} candidatos")
        
        return prospects_data, vagas_data, applicants_data
    
    def extrair_dados_basicos(self, prospects_data, vagas_data, applicants_data):
        """Extrai dados básicos para análise"""
        print("\\n📊 EXTRAINDO DADOS BÁSICOS...")
        
        dados = []
        
        for codigo_vaga, vaga_info in prospects_data.items():
            titulo_vaga = vaga_info.get('titulo', '')
            modalidade = vaga_info.get('modalidade', '')
            
            for prospect in vaga_info.get('prospects', []):
                codigo_candidato = prospect.get('codigo', '')
                situacao = prospect.get('situacao_candidado', '')
                data_candidatura = prospect.get('data_candidatura', '')
                recrutador = prospect.get('recrutador', '')
                
                # Buscar dados do candidato
                candidato_info = applicants_data.get(codigo_candidato, {})
                infos_pessoais = candidato_info.get('informacoes_pessoais', {})
                infos_profissionais = candidato_info.get('informacoes_profissionais', {})
                formacao = candidato_info.get('formacao_idiomas', {})
                
                # Buscar dados da vaga
                vaga_detalhes = vagas_data.get(codigo_vaga, {})
                info_basicas = vaga_detalhes.get('informacoes_basicas', {}) if isinstance(vaga_detalhes, dict) else {}
                
                registro = {
                    'codigo_vaga': codigo_vaga,
                    'codigo_candidato': codigo_candidato,
                    'situacao_candidado': situacao,
                    'data_candidatura': data_candidatura,
                    'recrutador': recrutador,
                    'titulo_vaga': titulo_vaga,
                    'modalidade': modalidade,
                    
                    # Dados do candidato
                    'sexo': infos_pessoais.get('sexo', ''),
                    'nivel_academico': formacao.get('nivel_academico', ''),
                    'nivel_ingles': formacao.get('nivel_ingles', ''),
                    'nivel_profissional': infos_profissionais.get('nivel_profissional', ''),
                    'area_atuacao': infos_profissionais.get('area_atuacao', ''),
                    'cv_texto': candidato_info.get('cv_pt', ''),
                    
                    # Dados da vaga
                    'cliente': info_basicas.get('cliente', ''),
                    'tipo_contratacao': info_basicas.get('tipo_contratacao', ''),
                    'cidade_vaga': info_basicas.get('cidade', ''),
                    'estado_vaga': info_basicas.get('estado', '')
                }
                
                dados.append(registro)
        
        df = pd.DataFrame(dados)
        
        # Criar clusters
        df['cluster'] = df['situacao_candidado'].map(self.clusters_mapping)
        df['cluster'] = df['cluster'].fillna(2)  # Em processo por padrão
        
        print(f"✓ Dataset criado: {len(df)} registros")
        print(f"✓ Colunas: {len(df.columns)}")
        
        return df
    
    def analisar_distribuicao_clusters(self, df):
        """Analisa distribuição dos 3 clusters"""
        print("\\n=== ANÁLISE DE DISTRIBUIÇÃO DOS 3 CLUSTERS ===")
        
        # Distribuição geral
        distribuicao = df['cluster'].value_counts().sort_index()
        total = len(df)
        
        print("\\n📊 DISTRIBUIÇÃO DOS CLUSTERS:")
        for cluster_id, count in distribuicao.items():
            pct = (count / total) * 100
            info = self.clusters_info[cluster_id]
            print(f"   {info['emoji']} Cluster {cluster_id} - {info['nome']}: {count:,} ({pct:.1f}%)")
        
        # Salvar insights
        self.insights['distribuicao_clusters'] = {
            'total_registros': int(total),
            'cluster_0_sucesso': {'count': int(distribuicao.get(0, 0)), 'pct': float((distribuicao.get(0, 0) / total) * 100)},
            'cluster_1_sairam': {'count': int(distribuicao.get(1, 0)), 'pct': float((distribuicao.get(1, 0) / total) * 100)},
            'cluster_2_processo': {'count': int(distribuicao.get(2, 0)), 'pct': float((distribuicao.get(2, 0) / total) * 100)}
        }
        
        # Visualização
        plt.figure(figsize=(15, 6))
        
        # Gráfico de barras
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(3), [distribuicao.get(i, 0) for i in range(3)], 
                      color=self.cores_clusters, alpha=0.8)
        
        plt.title('Distribuição dos 3 Clusters', fontsize=14, fontweight='bold')
        plt.xlabel('Clusters')
        plt.ylabel('Número de Candidatos')
        
        # Labels personalizadas
        labels = [f"{self.clusters_info[i]['emoji']} Cluster {i}\\n{self.clusters_info[i]['nome']}" for i in range(3)]
        plt.xticks(range(3), labels, rotation=0, ha='center')
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            pct = (height / total) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}\\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        
        # Gráfico de pizza
        plt.subplot(1, 2, 2)
        sizes = [distribuicao.get(i, 0) for i in range(3)]
        labels_pizza = [f"{self.clusters_info[i]['emoji']} {self.clusters_info[i]['nome']}\\n{sizes[i]:,} ({(sizes[i]/total)*100:.1f}%)" for i in range(3)]
        
        plt.pie(sizes, labels=labels_pizza, colors=self.cores_clusters, autopct='',
                startangle=90, textprops={'fontsize': 9})
        plt.title('Proporção dos Clusters', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'distribuicao_3_clusters.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Visualização salva: distribuicao_3_clusters.png")
    
    def analisar_situacoes_detalhadas(self, df):
        """Analisa situações detalhadas por cluster"""
        print("\\n=== ANÁLISE DETALHADA DAS SITUAÇÕES ===")
        
        # Situações por cluster
        situacoes_por_cluster = {}
        
        for cluster_id in range(3):
            df_cluster = df[df['cluster'] == cluster_id]
            situacoes = df_cluster['situacao_candidado'].value_counts()
            
            info = self.clusters_info[cluster_id]
            print(f"\\n{info['emoji']} CLUSTER {cluster_id} - {info['nome']}:")
            
            situacoes_dict = {}
            for situacao, count in situacoes.items():
                pct = (count / len(df_cluster)) * 100
                print(f"   • {situacao}: {count:,} ({pct:.1f}%)")
                situacoes_dict[situacao] = {'count': int(count), 'pct': float(pct)}
            
            situacoes_por_cluster[f'cluster_{cluster_id}'] = situacoes_dict
        
        self.insights['situacoes_por_cluster'] = situacoes_por_cluster
        
        # Visualização das principais situações
        plt.figure(figsize=(16, 10))
        
        for i, cluster_id in enumerate(range(3)):
            plt.subplot(2, 2, i+1)
            
            df_cluster = df[df['cluster'] == cluster_id]
            situacoes = df_cluster['situacao_candidado'].value_counts().head(8)
            
            info = self.clusters_info[cluster_id]
            
            bars = plt.barh(range(len(situacoes)), situacoes.values, 
                           color=info['cor'], alpha=0.7)
            
            plt.title(f"{info['emoji']} Cluster {cluster_id}\\n{info['nome']}", 
                     fontsize=12, fontweight='bold')
            plt.xlabel('Número de Candidatos')
            
            # Labels das situações (truncadas se muito longas)
            labels = [s[:30] + '...' if len(s) > 30 else s for s in situacoes.index]
            plt.yticks(range(len(situacoes)), labels)
            
            # Adicionar valores
            for j, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                        f'{int(width):,}', ha='left', va='center', fontsize=9)
            
            plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'situacoes_por_cluster.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Visualização salva: situacoes_por_cluster.png")
    
    def analisar_performance_recrutadores(self, df):
        """Analisa performance dos recrutadores por cluster"""
        print("\\n=== ANÁLISE DE PERFORMANCE DOS RECRUTADORES ===")
        
        # Performance por recrutador
        recrutador_stats = df.groupby('recrutador').agg({
            'cluster': ['count', lambda x: (x == 0).sum(), lambda x: (x == 1).sum(), lambda x: (x == 2).sum()]
        }).round(3)
        
        recrutador_stats.columns = ['total', 'sucessos', 'sairam', 'processo']
        recrutador_stats['taxa_sucesso'] = recrutador_stats['sucessos'] / recrutador_stats['total']
        recrutador_stats['taxa_saida'] = recrutador_stats['sairam'] / recrutador_stats['total']
        recrutador_stats['taxa_processo'] = recrutador_stats['processo'] / recrutador_stats['total']
        
        # Filtrar recrutadores com pelo menos 50 candidatos
        recrutador_stats = recrutador_stats[recrutador_stats['total'] >= 50].sort_values('taxa_sucesso', ascending=False)
        
        print(f"\\n📊 TOP 10 RECRUTADORES (com ≥50 candidatos):")
        print("Recrutador | Total | Sucessos | Taxa Sucesso | Taxa Saída")
        print("-" * 60)
        
        top_recrutadores = {}
        for idx, (recrutador, stats) in enumerate(recrutador_stats.head(10).iterrows()):
            print(f"{recrutador[:20]:20} | {stats['total']:5.0f} | {stats['sucessos']:8.0f} | {stats['taxa_sucesso']:11.1%} | {stats['taxa_saida']:9.1%}")
            
            top_recrutadores[recrutador] = {
                'total': int(stats['total']),
                'sucessos': int(stats['sucessos']),
                'taxa_sucesso': float(stats['taxa_sucesso']),
                'taxa_saida': float(stats['taxa_saida'])
            }
        
        self.insights['performance_recrutadores'] = {
            'total_recrutadores': len(recrutador_stats),
            'top_10': top_recrutadores,
            'media_taxa_sucesso': float(recrutador_stats['taxa_sucesso'].mean()),
            'media_taxa_saida': float(recrutador_stats['taxa_saida'].mean())
        }
        
        # Visualização
        plt.figure(figsize=(15, 8))
        
        # Top 15 recrutadores
        top_15 = recrutador_stats.head(15)
        
        x = range(len(top_15))
        width = 0.25
        
        plt.bar([i - width for i in x], top_15['taxa_sucesso'], width, 
               label='✅ Taxa Sucesso', color='#28a745', alpha=0.8)
        plt.bar(x, top_15['taxa_saida'], width, 
               label='⚠️ Taxa Saída', color='#ffc107', alpha=0.8)
        plt.bar([i + width for i in x], top_15['taxa_processo'], width, 
               label='🕐 Taxa Processo', color='#17a2b8', alpha=0.8)
        
        plt.title('Performance dos Top 15 Recrutadores', fontsize=14, fontweight='bold')
        plt.xlabel('Recrutadores')
        plt.ylabel('Taxa (%)')
        plt.legend()
        
        # Labels dos recrutadores (rotacionadas)
        labels = [r[:15] + '...' if len(r) > 15 else r for r in top_15.index]
        plt.xticks(x, labels, rotation=45, ha='right')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'performance_recrutadores.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Visualização salva: performance_recrutadores.png")
    
    def analisar_modalidades(self, df):
        """Analisa performance por modalidade"""
        print("\\n=== ANÁLISE POR MODALIDADE ===")
        
        # Performance por modalidade
        modalidade_stats = df.groupby('modalidade').agg({
            'cluster': ['count', lambda x: (x == 0).sum(), lambda x: (x == 1).sum(), lambda x: (x == 2).sum()]
        }).round(3)
        
        modalidade_stats.columns = ['total', 'sucessos', 'sairam', 'processo']
        modalidade_stats['taxa_sucesso'] = modalidade_stats['sucessos'] / modalidade_stats['total']
        modalidade_stats['taxa_saida'] = modalidade_stats['sairam'] / modalidade_stats['total']
        modalidade_stats['taxa_processo'] = modalidade_stats['processo'] / modalidade_stats['total']
        
        modalidade_stats = modalidade_stats.sort_values('taxa_sucesso', ascending=False)
        
        print(f"\\n📊 PERFORMANCE POR MODALIDADE:")
        print("Modalidade | Total | Sucessos | Taxa Sucesso | Taxa Saída")
        print("-" * 55)
        
        modalidades_info = {}
        for modalidade, stats in modalidade_stats.iterrows():
            print(f"{modalidade[:15]:15} | {stats['total']:5.0f} | {stats['sucessos']:8.0f} | {stats['taxa_sucesso']:11.1%} | {stats['taxa_saida']:9.1%}")
            
            modalidades_info[modalidade] = {
                'total': int(stats['total']),
                'sucessos': int(stats['sucessos']),
                'taxa_sucesso': float(stats['taxa_sucesso']),
                'taxa_saida': float(stats['taxa_saida'])
            }
        
        self.insights['performance_modalidades'] = modalidades_info
        
        # Visualização
        plt.figure(figsize=(12, 8))
        
        x = range(len(modalidade_stats))
        width = 0.25
        
        plt.bar([i - width for i in x], modalidade_stats['taxa_sucesso'], width, 
               label='✅ Taxa Sucesso', color='#28a745', alpha=0.8)
        plt.bar(x, modalidade_stats['taxa_saida'], width, 
               label='⚠️ Taxa Saída', color='#ffc107', alpha=0.8)
        plt.bar([i + width for i in x], modalidade_stats['taxa_processo'], width, 
               label='🕐 Taxa Processo', color='#17a2b8', alpha=0.8)
        
        plt.title('Performance por Modalidade', fontsize=14, fontweight='bold')
        plt.xlabel('Modalidades')
        plt.ylabel('Taxa (%)')
        plt.legend()
        
        plt.xticks(x, modalidade_stats.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'performance_modalidades.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Visualização salva: performance_modalidades.png")
    
    def analisar_tendencias_temporais(self, df):
        """Analisa tendências temporais"""
        print("\\n=== ANÁLISE DE TENDÊNCIAS TEMPORAIS ===")
        
        # Converter data
        df['data_candidatura'] = pd.to_datetime(df['data_candidatura'], errors='coerce')
        df_temporal = df.dropna(subset=['data_candidatura'])
        
        if len(df_temporal) == 0:
            print("⚠️ Não há dados temporais válidos")
            return
        
        # Análise mensal
        df_temporal['mes_ano'] = df_temporal['data_candidatura'].dt.to_period('M')
        
        temporal_stats = df_temporal.groupby(['mes_ano', 'cluster']).size().unstack(fill_value=0)
        
        # Calcular taxas
        temporal_stats['total'] = temporal_stats.sum(axis=1)
        temporal_stats['taxa_sucesso'] = temporal_stats.get(0, 0) / temporal_stats['total']
        temporal_stats['taxa_saida'] = temporal_stats.get(1, 0) / temporal_stats['total']
        temporal_stats['taxa_processo'] = temporal_stats.get(2, 0) / temporal_stats['total']
        
        print(f"\\n📊 TENDÊNCIAS MENSAIS (últimos 12 meses):")
        print("Mês/Ano | Total | Taxa Sucesso | Taxa Saída | Taxa Processo")
        print("-" * 60)
        
        tendencias_mensais = {}
        for mes_ano, stats in temporal_stats.tail(12).iterrows():
            print(f"{str(mes_ano):7} | {stats['total']:5.0f} | {stats['taxa_sucesso']:11.1%} | {stats['taxa_saida']:9.1%} | {stats['taxa_processo']:12.1%}")
            
            tendencias_mensais[str(mes_ano)] = {
                'total': int(stats['total']),
                'taxa_sucesso': float(stats['taxa_sucesso']),
                'taxa_saida': float(stats['taxa_saida']),
                'taxa_processo': float(stats['taxa_processo'])
            }
        
        self.insights['tendencias_temporais'] = tendencias_mensais
        
        # Visualização
        plt.figure(figsize=(15, 8))
        
        # Últimos 12 meses
        dados_recentes = temporal_stats.tail(12)
        
        plt.subplot(2, 1, 1)
        plt.plot(dados_recentes.index.astype(str), dados_recentes['taxa_sucesso'], 
                marker='o', linewidth=2, color='#28a745', label='✅ Taxa Sucesso')
        plt.plot(dados_recentes.index.astype(str), dados_recentes['taxa_saida'], 
                marker='s', linewidth=2, color='#ffc107', label='⚠️ Taxa Saída')
        plt.plot(dados_recentes.index.astype(str), dados_recentes['taxa_processo'], 
                marker='^', linewidth=2, color='#17a2b8', label='🕐 Taxa Processo')
        
        plt.title('Tendências Temporais - Taxas por Cluster', fontsize=14, fontweight='bold')
        plt.ylabel('Taxa (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.bar(dados_recentes.index.astype(str), dados_recentes['total'], 
               color='#6c757d', alpha=0.7)
        plt.title('Volume Total de Candidatos por Mês', fontsize=12, fontweight='bold')
        plt.ylabel('Número de Candidatos')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'tendencias_temporais.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Visualização salva: tendencias_temporais.png")
    
    def analisar_caracteristicas_demograficas(self, df):
        """Analisa características demográficas por cluster"""
        print("\\n=== ANÁLISE DEMOGRÁFICA POR CLUSTER ===")
        
        # Análise por gênero
        if 'sexo' in df.columns and not df['sexo'].isna().all():
            print("\\n👥 DISTRIBUIÇÃO POR GÊNERO:")
            genero_cluster = pd.crosstab(df['sexo'], df['cluster'], normalize='columns') * 100
            
            for cluster_id in range(3):
                if cluster_id in genero_cluster.columns:
                    info = self.clusters_info[cluster_id]
                    print(f"\\n{info['emoji']} {info['nome']}:")
                    for genero, pct in genero_cluster[cluster_id].items():
                        if pd.notna(genero) and genero.strip():
                            print(f"   {genero}: {pct:.1f}%")
        
        # Análise por nível acadêmico
        if 'nivel_academico' in df.columns and not df['nivel_academico'].isna().all():
            print("\\n🎓 DISTRIBUIÇÃO POR NÍVEL ACADÊMICO:")
            academico_cluster = pd.crosstab(df['nivel_academico'], df['cluster'], normalize='columns') * 100
            
            for cluster_id in range(3):
                if cluster_id in academico_cluster.columns:
                    info = self.clusters_info[cluster_id]
                    print(f"\\n{info['emoji']} {info['nome']}:")
                    for nivel, pct in academico_cluster[cluster_id].head(5).items():
                        if pd.notna(nivel) and nivel.strip():
                            print(f"   {nivel}: {pct:.1f}%")
        
        # Análise por nível de inglês
        if 'nivel_ingles' in df.columns and not df['nivel_ingles'].isna().all():
            print("\\n🌐 DISTRIBUIÇÃO POR NÍVEL DE INGLÊS:")
            ingles_cluster = pd.crosstab(df['nivel_ingles'], df['cluster'], normalize='columns') * 100
            
            for cluster_id in range(3):
                if cluster_id in ingles_cluster.columns:
                    info = self.clusters_info[cluster_id]
                    print(f"\\n{info['emoji']} {info['nome']}:")
                    for nivel, pct in ingles_cluster[cluster_id].head(5).items():
                        if pd.notna(nivel) and nivel.strip():
                            print(f"   {nivel}: {pct:.1f}%")
    
    def gerar_insights_estrategicos(self, df):
        """Gera insights estratégicos para o negócio"""
        print("\\n=== INSIGHTS ESTRATÉGICOS ===")
        
        insights_estrategicos = []
        
        # Insight 1: Taxa de conversão geral
        taxa_sucesso_geral = (df['cluster'] == 0).mean()
        if taxa_sucesso_geral < 0.10:
            insights_estrategicos.append({
                'tipo': 'conversao',
                'prioridade': 'alta',
                'titulo': 'Taxa de Conversão Baixa',
                'descricao': f'Taxa de sucesso de apenas {taxa_sucesso_geral:.1%}. Oportunidade de melhoria no funil.',
                'acao_recomendada': 'Analisar gargalos no processo e otimizar etapas de maior perda.'
            })
        
        # Insight 2: Variação entre recrutadores
        if 'performance_recrutadores' in self.insights:
            recrutadores = self.insights['performance_recrutadores']['top_10']
            if recrutadores:
                taxas = [r['taxa_sucesso'] for r in recrutadores.values()]
                variacao = max(taxas) - min(taxas)
                if variacao > 0.15:  # 15% de diferença
                    insights_estrategicos.append({
                        'tipo': 'recrutadores',
                        'prioridade': 'media',
                        'titulo': 'Grande Variação entre Recrutadores',
                        'descricao': f'Diferença de {variacao:.1%} entre melhor e pior recrutador.',
                        'acao_recomendada': 'Padronizar processos e treinar recrutadores com menor performance.'
                    })
        
        # Insight 3: Modalidades mais eficazes
        if 'performance_modalidades' in self.insights:
            modalidades = self.insights['performance_modalidades']
            melhor_modalidade = max(modalidades.items(), key=lambda x: x[1]['taxa_sucesso'])
            if melhor_modalidade[1]['taxa_sucesso'] > taxa_sucesso_geral * 1.5:
                insights_estrategicos.append({
                    'tipo': 'modalidades',
                    'prioridade': 'media',
                    'titulo': 'Modalidade de Alto Desempenho Identificada',
                    'descricao': f'{melhor_modalidade[0]} tem taxa de sucesso de {melhor_modalidade[1]["taxa_sucesso"]:.1%}.',
                    'acao_recomendada': 'Expandir investimento nesta modalidade e aplicar práticas em outras.'
                })
        
        # Insight 4: Volume de candidatos em processo
        candidatos_processo = (df['cluster'] == 2).sum()
        if candidatos_processo > len(df) * 0.6:  # Mais de 60% em processo
            insights_estrategicos.append({
                'tipo': 'processo',
                'prioridade': 'alta',
                'titulo': 'Alto Volume de Candidatos em Processo',
                'descricao': f'{candidatos_processo:,} candidatos ({(candidatos_processo/len(df)):.1%}) ainda em processo.',
                'acao_recomendada': 'Acelerar processo de decisão e implementar sistema de priorização.'
            })
        
        # Salvar insights
        self.insights['insights_estrategicos'] = insights_estrategicos
        
        print("\\n🎯 INSIGHTS ESTRATÉGICOS IDENTIFICADOS:")
        for i, insight in enumerate(insights_estrategicos, 1):
            prioridade_emoji = '🔴' if insight['prioridade'] == 'alta' else '🟡'
            print(f"\\n{i}. {prioridade_emoji} {insight['titulo']} ({insight['prioridade'].upper()})")
            print(f"   📊 {insight['descricao']}")
            print(f"   💡 {insight['acao_recomendada']}")
    
    def salvar_relatorio_completo(self):
        """Salva relatório completo em JSON"""
        print("\\n💾 SALVANDO RELATÓRIO COMPLETO...")
        
        # Adicionar metadados
        self.insights['metadata'] = {
            'data_analise': datetime.now().isoformat(),
            'versao': '3_clusters_final',
            'clusters_definicao': self.clusters_info
        }
        
        # Salvar JSON
        relatorio_path = os.path.join(self.reports_dir, 'analise_exploratoria_3_clusters.json')
        with open(relatorio_path, 'w', encoding='utf-8') as f:
            json.dump(self.insights, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ Relatório salvo: {relatorio_path}")
        
        return relatorio_path
    
    def executar_analise_completa(self):
        """Executa análise exploratória completa"""
        print("🎯 INICIANDO ANÁLISE EXPLORATÓRIA - 3 CLUSTERS ESPECÍFICOS")
        print("=" * 70)
        
        # 1. Carregar dados
        prospects_data, vagas_data, applicants_data = self.carregar_dados()
        
        # 2. Extrair dados básicos
        df = self.extrair_dados_basicos(prospects_data, vagas_data, applicants_data)
        
        # 3. Análise de distribuição dos clusters
        self.analisar_distribuicao_clusters(df)
        
        # 4. Análise detalhada das situações
        self.analisar_situacoes_detalhadas(df)
        
        # 5. Performance dos recrutadores
        self.analisar_performance_recrutadores(df)
        
        # 6. Análise por modalidade
        self.analisar_modalidades(df)
        
        # 7. Tendências temporais
        self.analisar_tendencias_temporais(df)
        
        # 8. Características demográficas
        self.analisar_caracteristicas_demograficas(df)
        
        # 9. Insights estratégicos
        self.gerar_insights_estrategicos(df)
        
        # 10. Salvar relatório
        relatorio_path = self.salvar_relatorio_completo()
        
        print("\\n🎉 ANÁLISE EXPLORATÓRIA DOS 3 CLUSTERS CONCLUÍDA!")
        print("=" * 70)
        print(f"📊 Total de registros analisados: {len(df):,}")
        print(f"📁 Visualizações salvas em: {self.visualizations_dir}")
        print(f"📄 Relatório salvo em: {relatorio_path}")
        
        # Resumo final
        distribuicao = df['cluster'].value_counts().sort_index()
        total = len(df)
        
        print(f"\\n📊 RESUMO FINAL:")
        for cluster_id, count in distribuicao.items():
            pct = (count / total) * 100
            info = self.clusters_info[cluster_id]
            print(f"   {info['emoji']} Cluster {cluster_id}: {count:,} candidatos ({pct:.1f}%)")
        
        return df, self.insights

def main():
    """Função principal"""
    analisador = AnalisadorExploratorio3Clusters()
    df, insights = analisador.executar_analise_completa()
    
    print(f"\\n✅ Análise concluída com sucesso!")
    print(f"   - Registros: {len(df):,}")
    print(f"   - Insights: {len(insights)} categorias")
    print(f"   - Visualizações: 5 gráficos gerados")

if __name__ == "__main__":
    main()

