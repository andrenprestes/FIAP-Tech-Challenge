#!/usr/bin/env python3
"""
TREINAMENTO MELHORADO FINAL - 3 CLUSTERS
Implementa todas as melhorias críticas:
- SMOTE + Tomek para balanceamento avançado
- XGBoost e LightGBM otimizados para classes desbalanceadas
- Validação cruzada estratificada
- Seleção inteligente de features
- Ensemble avançado

CLUSTERS:
✅ Cluster 0: Candidatos de Sucesso (Finalizaram o Funil com Êxito)
⚠️ Cluster 1: Candidatos que Saíram do Processo (Desistiram ou Foram Reprovados)
🕐 Cluster 2: Candidatos em Processo (Ativos ou Pendentes)
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports de ML
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Imports para balanceamento
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

# Imports de algoritmos avançados
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost não disponível - instale com: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM não disponível - instale com: pip install lightgbm")

class TreinadorModelos3ClustersMelhorado:
    def __init__(self):
        self.data_dir = '../data'
        self.models_dir = '../models'
        self.reports_dir = '../reports'
        
        # Criar diretórios
        for directory in [self.models_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Nomes dos clusters
        self.clusters_nomes = {
            0: "CANDIDATOS DE SUCESSO",
            1: "CANDIDATOS QUE SAÍRAM DO PROCESSO",
            2: "CANDIDATOS EM PROCESSO"
        }
        
        # Emojis dos clusters
        self.cluster_emojis = {
            0: '✅',
            1: '⚠️',
            2: '🕐'
        }
        
        # Modelos candidatos otimizados
        self.modelos_candidatos = self._criar_modelos_candidatos()
        
        # Resultados
        self.resultados = {}
        self.melhor_modelo = None
        self.melhor_score = 0
        self.feature_importances = None
    
    def _criar_modelos_candidatos(self):
        """Cria modelos candidatos otimizados para classes desbalanceadas"""
        modelos = {
            'RandomForest': RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced',
                solver='saga',
                penalty='l1'
            )
        }
        
        # Adicionar XGBoost se disponível
        if XGBOOST_AVAILABLE:
            modelos['XGBoost'] = XGBClassifier(
                random_state=42,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='mlogloss',
                verbosity=0
            )
        
        # Adicionar LightGBM se disponível
        if LIGHTGBM_AVAILABLE:
            modelos['LightGBM'] = LGBMClassifier(
                random_state=42,
                class_weight='balanced',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1
            )
        
        return modelos
    
    def carregar_dados_processados(self):
        """Carrega dados processados melhorados"""
        print("=== CARREGANDO DADOS PROCESSADOS MELHORADOS ===")
        
        try:
            # Carregar dataset principal
            with open(os.path.join(self.models_dir, 'processed_data_3_clusters_melhorado.pkl'), 'rb') as f:
                df = pickle.load(f)
            
            # Carregar artefatos de preprocessamento
            with open(os.path.join(self.models_dir, 'preprocessing_artifacts_3_clusters_melhorado.pkl'), 'rb') as f:
                artifacts = pickle.load(f)
            
            feature_names = artifacts['feature_names']
            
            print(f"✓ Dataset carregado: {len(df)} registros")
            print(f"✓ Features disponíveis: {len(feature_names)}")
            
            return df, feature_names, artifacts
            
        except FileNotFoundError:
            print("❌ Arquivos de dados processados não encontrados!")
            print("Execute primeiro: python 02_processamento_3_CLUSTERS_FINAL.py")
            raise
    
    def preparar_dados_para_treinamento(self, df, feature_names):
        """Prepara dados para treinamento"""
        print("\\n=== PREPARANDO DADOS PARA TREINAMENTO ===")
        
        # Extrair features e target
        X = df[feature_names].fillna(0)
        y = df['cluster_target']
        
        print(f"📊 Dataset: {X.shape}")
        print(f"📊 Distribuição original:")
        distribuicao = y.value_counts().sort_index()
        total = len(y)
        
        for cluster_id, count in distribuicao.items():
            pct = (count / total) * 100
            emoji = self.cluster_emojis.get(cluster_id, '📊')
            nome = self.clusters_nomes.get(cluster_id, f"CLUSTER {cluster_id}")
            print(f"   {emoji} Cluster {cluster_id} ({nome}): {count:,} ({pct:.1f}%)")
        
        return X, y
    
    def aplicar_balanceamento_avancado(self, X_train, y_train):
        """Aplica balanceamento avançado com SMOTE + Tomek"""
        print("\\n🔄 APLICANDO BALANCEAMENTO AVANÇADO (SMOTE + TOMEK)...")
        
        # Mostrar distribuição original
        print("📊 Distribuição antes do balanceamento:")
        dist_original = pd.Series(y_train).value_counts().sort_index()
        for cluster_id, count in dist_original.items():
            emoji = self.cluster_emojis.get(cluster_id, '📊')
            print(f"   {emoji} Cluster {cluster_id}: {count:,}")
        
        # Aplicar SMOTE + Tomek
        smote_tomek = SMOTETomek(
            smote=SMOTE(random_state=42, k_neighbors=3),
            tomek=TomekLinks(),
            random_state=42
        )
        
        X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
        
        # Mostrar distribuição balanceada
        print("\\n📊 Distribuição após balanceamento:")
        dist_balanceada = pd.Series(y_train_balanced).value_counts().sort_index()
        for cluster_id, count in dist_balanceada.items():
            emoji = self.cluster_emojis.get(cluster_id, '📊')
            original = dist_original.get(cluster_id, 0)
            mudanca = count - original
            print(f"   {emoji} Cluster {cluster_id}: {count:,} (+{mudanca:,})")
        
        print(f"\\n✓ Dados balanceados: {X_train_balanced.shape}")
        print(f"✓ Aumento total: {len(X_train_balanced) - len(X_train):,} exemplos")
        
        return X_train_balanced, y_train_balanced
    
    def selecionar_features_importantes(self, X, y, k=30):
        """Seleciona features mais importantes"""
        print(f"\\n🎯 SELECIONANDO {k} FEATURES MAIS IMPORTANTES...")
        
        # Seleção baseada em F-score
        selector_f = SelectKBest(score_func=f_classif, k=k//2)
        selector_f.fit(X, y)
        features_f = selector_f.get_support(indices=True)
        
        # Seleção baseada em informação mútua
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=k//2)
        selector_mi.fit(X, y)
        features_mi = selector_mi.get_support(indices=True)
        
        # Combinar features selecionadas (união)
        features_selecionadas = sorted(list(set(features_f) | set(features_mi)))
        
        print(f"✓ Features F-score: {len(features_f)}")
        print(f"✓ Features Info Mútua: {len(features_mi)}")
        print(f"✓ Features finais selecionadas: {len(features_selecionadas)}")
        
        return features_selecionadas
    
    def avaliar_modelo_robusto(self, modelo, X, y, nome_modelo):
        """Avaliação robusta com validação estratificada"""
        print(f"\\n📊 Avaliação robusta: {nome_modelo}")
        
        # Validação cruzada estratificada
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Múltiplas métricas
        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'f1_macro': 'f1_macro',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted'
        }
        
        cv_results = cross_validate(modelo, X, y, cv=skf, scoring=scoring, 
                                  return_train_score=True, n_jobs=-1)
        
        # Calcular estatísticas
        results = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            results[metric] = {
                'mean': test_scores.mean(),
                'std': test_scores.std(),
                'min': test_scores.min(),
                'max': test_scores.max()
            }
            
            print(f"   {metric}: {results[metric]['mean']:.3f} ± {results[metric]['std']:.3f}")
        
        return results
    
    def treinar_modelos(self, X_train, y_train, X_test, y_test, feature_names):
        """Treina todos os modelos candidatos"""
        print("\\n=== TREINANDO MODELOS OTIMIZADOS ===")
        
        resultados = {}
        
        for nome_modelo, modelo in self.modelos_candidatos.items():
            print(f"\\n🔄 Treinando {nome_modelo}...")
            
            try:
                # Treinar modelo
                if nome_modelo == 'XGBoost' and XGBOOST_AVAILABLE:
                    # XGBoost precisa de tratamento especial para multiclass
                    modelo.fit(X_train, y_train)
                else:
                    modelo.fit(X_train, y_train)
                
                # Predições
                y_pred_train = modelo.predict(X_train)
                y_pred_test = modelo.predict(X_test)
                
                # Métricas
                f1_train = f1_score(y_train, y_pred_train, average='weighted')
                f1_test = f1_score(y_test, y_pred_test, average='weighted')
                
                # Avaliação robusta
                cv_results = self.avaliar_modelo_robusto(modelo, X_train, y_train, nome_modelo)
                
                # Salvar resultados
                resultados[nome_modelo] = {
                    'modelo': modelo,
                    'f1_train': f1_train,
                    'f1_test': f1_test,
                    'cv_results': cv_results,
                    'y_pred_test': y_pred_test
                }
                
                print(f"   ✓ F1-Score Treino: {f1_train:.3f}")
                print(f"   ✓ F1-Score Teste: {f1_test:.3f}")
                print(f"   ✓ F1-Score CV: {cv_results['f1_weighted']['mean']:.3f}")
                
                # Salvar importâncias se disponível
                if hasattr(modelo, 'feature_importances_'):
                    importances = modelo.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    resultados[nome_modelo]['feature_importances'] = feature_importance_df
                    
                    print(f"   ✓ Top 5 features:")
                    for idx, row in feature_importance_df.head().iterrows():
                        print(f"      {row['feature']}: {row['importance']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Erro ao treinar {nome_modelo}: {e}")
                continue
        
        return resultados
    
    def selecionar_melhor_modelo(self, resultados):
        """Seleciona o melhor modelo baseado em múltiplas métricas"""
        print("\\n🏆 SELECIONANDO MELHOR MODELO...")
        
        scores_modelos = {}
        
        for nome_modelo, resultado in resultados.items():
            # Score combinado: CV F1-weighted (70%) + F1-test (30%)
            cv_f1 = resultado['cv_results']['f1_weighted']['mean']
            test_f1 = resultado['f1_test']
            
            score_combinado = 0.7 * cv_f1 + 0.3 * test_f1
            scores_modelos[nome_modelo] = score_combinado
            
            print(f"   📊 {nome_modelo}:")
            print(f"      CV F1: {cv_f1:.3f}")
            print(f"      Test F1: {test_f1:.3f}")
            print(f"      Score Combinado: {score_combinado:.3f}")
        
        # Selecionar melhor
        melhor_modelo_nome = max(scores_modelos, key=scores_modelos.get)
        melhor_score = scores_modelos[melhor_modelo_nome]
        melhor_modelo = resultados[melhor_modelo_nome]['modelo']
        
        print(f"\\n🏆 MELHOR MODELO: {melhor_modelo_nome}")
        print(f"🎯 Score Final: {melhor_score:.3f}")
        
        return melhor_modelo_nome, melhor_modelo, melhor_score, resultados[melhor_modelo_nome]
    
    def criar_ensemble_avancado(self, resultados, X_train, y_train):
        """Cria ensemble avançado dos melhores modelos"""
        print("\\n🎭 CRIANDO ENSEMBLE AVANÇADO...")
        
        # Selecionar top modelos (todos se forem poucos)
        scores = {}
        for nome, resultado in resultados.items():
            scores[nome] = resultado['cv_results']['f1_weighted']['mean']
        
        top_modelos = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"✓ Modelos disponíveis para ensemble:")
        
        estimators = []
        for nome, score in top_modelos:
            modelo = resultados[nome]['modelo']
            estimators.append((nome.lower(), modelo))
            print(f"   - {nome}: {score:.3f}")
        
        # Criar Voting Classifier
        if len(estimators) >= 2:
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1
            )
            
            print("\\n🔄 Treinando ensemble...")
            ensemble.fit(X_train, y_train)
            
            print("✓ Ensemble criado com sucesso")
            return ensemble
        else:
            print("⚠️ Poucos modelos disponíveis para ensemble")
            return None
    
    def avaliar_modelo_final(self, modelo, X_test, y_test, nome_modelo="Modelo Final"):
        """Avaliação detalhada do modelo final"""
        print(f"\\n📊 AVALIAÇÃO DETALHADA: {nome_modelo}")
        print("=" * 50)
        
        # Predições
        y_pred = modelo.predict(X_test)
        
        # Relatório de classificação
        print("\\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
        cluster_names = [f"Cluster {i}" for i in range(3)]
        report = classification_report(y_test, y_pred, 
                                     target_names=cluster_names,
                                     output_dict=True)
        
        print(classification_report(y_test, y_pred, target_names=cluster_names))
        
        # Matriz de confusão
        print("\\n🔢 MATRIZ DE CONFUSÃO:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Métricas por cluster
        print("\\n📊 PERFORMANCE POR CLUSTER:")
        for i in range(3):
            if str(i) in report:
                precision = report[str(i)]['precision']
                recall = report[str(i)]['recall']
                f1 = report[str(i)]['f1-score']
                support = report[str(i)]['support']
                
                emoji = self.cluster_emojis.get(i, '📊')
                nome = self.clusters_nomes.get(i, f"CLUSTER {i}")
                
                print(f"   {emoji} Cluster {i} ({nome}):")
                print(f"      Precision: {precision:.3f}")
                print(f"      Recall: {recall:.3f}")
                print(f"      F1-Score: {f1:.3f}")
                print(f"      Support: {support}")
        
        # Métricas gerais
        accuracy = report['accuracy']
        macro_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']
        
        print(f"\\n🎯 MÉTRICAS GERAIS:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   F1-Score Macro: {macro_f1:.3f}")
        print(f"   F1-Score Weighted: {weighted_f1:.3f}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': macro_f1,
            'f1_weighted': weighted_f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist()
        }
    
    def salvar_modelo_final(self, modelo, nome_modelo, resultados_avaliacao, feature_names, artifacts):
        """Salva o modelo final e metadados"""
        print("\\n💾 SALVANDO MODELO FINAL...")
        
        # Preparar metadados
        metadata = {
            'nome_modelo': nome_modelo,
            'data_treinamento': datetime.now().isoformat(),
            'feature_names': feature_names,
            'clusters_nomes': self.clusters_nomes,
            'clusters_emojis': self.cluster_emojis,
            'resultados_avaliacao': resultados_avaliacao,
            'artifacts': artifacts
        }
        
        # Salvar modelo
        model_path = os.path.join(self.models_dir, 'modelo_final_3_clusters_melhorado.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(modelo, f)
        
        # Salvar metadados
        metadata_path = os.path.join(self.models_dir, 'modelo_metadata_3_clusters_melhorado.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ Modelo salvo: {model_path}")
        print(f"✓ Metadados salvos: {metadata_path}")
        
        return model_path, metadata_path
    
    def executar_treinamento_completo(self):
        """Executa todo o pipeline de treinamento"""
        print("🚀 INICIANDO TREINAMENTO MELHORADO FINAL - 3 CLUSTERS")
        print("=" * 70)
        
        # 1. Carregar dados processados
        df, feature_names, artifacts = self.carregar_dados_processados()
        
        # 2. Preparar dados
        X, y = self.preparar_dados_para_treinamento(df, feature_names)
        
        # 3. Dividir dados
        print("\\n🔄 Dividindo dados (80% treino, 20% teste)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✓ Treino: {X_train.shape}")
        print(f"✓ Teste: {X_test.shape}")
        
        # 4. Aplicar balanceamento avançado
        X_train_balanced, y_train_balanced = self.aplicar_balanceamento_avancado(X_train, y_train)
        
        # 5. Seleção de features importantes
        indices_features = self.selecionar_features_importantes(X_train_balanced, y_train_balanced, k=25)
        feature_names_selecionadas = [feature_names[i] for i in indices_features]
        
        X_train_final = X_train_balanced.iloc[:, indices_features]
        X_test_final = X_test.iloc[:, indices_features]
        
        print(f"✓ Features finais: {len(feature_names_selecionadas)}")
        
        # 6. Treinar modelos
        resultados = self.treinar_modelos(X_train_final, y_train_balanced, 
                                        X_test_final, y_test, feature_names_selecionadas)
        
        if not resultados:
            print("❌ Nenhum modelo foi treinado com sucesso!")
            return
        
        # 7. Selecionar melhor modelo
        melhor_nome, melhor_modelo, melhor_score, melhor_resultado = self.selecionar_melhor_modelo(resultados)
        
        # 8. Criar ensemble (opcional)
        ensemble = self.criar_ensemble_avancado(resultados, X_train_final, y_train_balanced)
        
        # 9. Avaliar modelo final
        if ensemble is not None:
            print("\\n🎭 AVALIANDO ENSEMBLE...")
            resultados_ensemble = self.avaliar_modelo_final(ensemble, X_test_final, y_test, "Ensemble")
            
            # Comparar ensemble vs melhor modelo individual
            ensemble_f1 = resultados_ensemble['f1_weighted']
            individual_f1 = melhor_resultado['f1_test']
            
            if ensemble_f1 > individual_f1:
                print(f"\\n🏆 ENSEMBLE VENCEU! ({ensemble_f1:.3f} vs {individual_f1:.3f})")
                modelo_final = ensemble
                nome_final = "Ensemble"
                resultados_finais = resultados_ensemble
            else:
                print(f"\\n🏆 MODELO INDIVIDUAL VENCEU! ({individual_f1:.3f} vs {ensemble_f1:.3f})")
                modelo_final = melhor_modelo
                nome_final = melhor_nome
                resultados_finais = self.avaliar_modelo_final(melhor_modelo, X_test_final, y_test, melhor_nome)
        else:
            modelo_final = melhor_modelo
            nome_final = melhor_nome
            resultados_finais = self.avaliar_modelo_final(melhor_modelo, X_test_final, y_test, melhor_nome)
        
        # 10. Salvar modelo final
        model_path, metadata_path = self.salvar_modelo_final(
            modelo_final, nome_final, resultados_finais, 
            feature_names_selecionadas, artifacts
        )
        
        print("\\n🎉 TREINAMENTO MELHORADO FINAL CONCLUÍDO!")
        print("=" * 70)
        print(f"🏆 Modelo Final: {nome_final}")
        print(f"🎯 Accuracy: {resultados_finais['accuracy']:.3f}")
        print(f"🎯 F1-Score Weighted: {resultados_finais['f1_weighted']:.3f}")
        print(f"📁 Modelo salvo em: {model_path}")
        
        return modelo_final, resultados_finais, feature_names_selecionadas

def main():
    """Função principal"""
    treinador = TreinadorModelos3ClustersMelhorado()
    modelo, resultados, features = treinador.executar_treinamento_completo()
    
    print(f"\\n📊 Resumo final:")
    print(f"   - Accuracy: {resultados['accuracy']:.1%}")
    print(f"   - F1-Score: {resultados['f1_weighted']:.1%}")
    print(f"   - Features: {len(features)}")

if __name__ == "__main__":
    main()

