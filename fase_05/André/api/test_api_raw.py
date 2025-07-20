#!/usr/bin/env python3
"""
TESTE COMPLETO DA API - 3 CLUSTERS ESPECÍFICOS
Script para testar todos os endpoints da API de classificação de candidatos
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

# Configurações
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30

class APITester:
    """Classe para testar a API de 3 clusters"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
        
    def log_result(self, test_name: str, success: bool, message: str, response_data: Any = None):
        """Registra resultado do teste"""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'response_data': response_data
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
        
        if response_data and not success:
            print(f"   Dados da resposta: {response_data}")
    
    def test_health_check(self):
        """Testa endpoint de health check"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                model_loaded = data.get('model_loaded', False)
                
                if model_loaded:
                    self.log_result(
                        "Health Check", 
                        True, 
                        f"API saudável, modelo carregado: {data.get('model_info', {}).get('algorithm', 'Unknown')}"
                    )
                else:
                    self.log_result(
                        "Health Check", 
                        False, 
                        "API respondeu mas modelo não está carregado",
                        data
                    )
            else:
                self.log_result(
                    "Health Check", 
                    False, 
                    f"Status code: {response.status_code}",
                    response.text
                )
                
        except Exception as e:
            self.log_result("Health Check", False, f"Erro de conexão: {e}")
    
    def test_root_endpoint(self):
        """Testa endpoint raiz"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                self.log_result(
                    "Root Endpoint", 
                    True, 
                    f"Versão: {data.get('version', 'N/A')}"
                )
            else:
                self.log_result(
                    "Root Endpoint", 
                    False, 
                    f"Status code: {response.status_code}",
                    response.text
                )
                
        except Exception as e:
            self.log_result("Root Endpoint", False, f"Erro: {e}")
    
    def test_clusters_info(self):
        """Testa endpoint de informações dos clusters"""
        try:
            response = self.session.get(f"{self.base_url}/clusters", timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                clusters_count = len(data)
                
                if clusters_count == 3:
                    cluster_names = [info.get('nome', 'N/A') for info in data.values()]
                    self.log_result(
                        "Clusters Info", 
                        True, 
                        f"3 clusters encontrados: {', '.join(cluster_names)}"
                    )
                else:
                    self.log_result(
                        "Clusters Info", 
                        False, 
                        f"Esperado 3 clusters, encontrado {clusters_count}",
                        data
                    )
            else:
                self.log_result(
                    "Clusters Info", 
                    False, 
                    f"Status code: {response.status_code}",
                    response.text
                )
                
        except Exception as e:
            self.log_result("Clusters Info", False, f"Erro: {e}")
    
    def test_model_info(self):
        """Testa endpoint de informações do modelo"""
        try:
            response = self.session.get(f"{self.base_url}/model/info", timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                algorithm = data.get('algorithm', 'Unknown')
                features_count = data.get('features_count', 0)
                
                self.log_result(
                    "Model Info", 
                    True, 
                    f"Algoritmo: {algorithm}, Features: {features_count}"
                )
            else:
                self.log_result(
                    "Model Info", 
                    False, 
                    f"Status code: {response.status_code}",
                    response.text
                )
                
        except Exception as e:
            self.log_result("Model Info", False, f"Erro: {e}")
    
    def test_single_prediction(self):
        """Testa predição individual"""
        
        # Candidato de teste - perfil de sucesso
        candidato_sucesso = {
            "codigo_candidato": "TEST_001",
            "sexo": "Masculino",
            "idade": 28,
            "nivel_academico": "Superior Completo",
            "nivel_ingles": "Avançado",
            "nivel_profissional": "Pleno",
            "area_atuacao": "Tecnologia",
            "cv_texto": "Desenvolvedor Python sênior com 5 anos de experiência em desenvolvimento web, especialista em Django e Flask, conhecimento avançado em SQL e JavaScript, experiência com metodologias ágeis",
            "data_candidatura": "2024-01-15",
            "recrutador": "Ana Silva",
            "codigo_vaga": "VAGA_001",
            "titulo_vaga": "Desenvolvedor Python Sênior",
            "modalidade": "CLT",
            "cliente": "Tech Corp",
            "tipo_contratacao": "CLT",
            "cidade_vaga": "São Paulo",
            "estado_vaga": "SP",
            "salario_minimo": 8000.0,
            "salario_maximo": 12000.0,
            "experiencia_minima": 3,
            "experiencia_maxima": 7
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict", 
                json=candidato_sucesso,
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success', False):
                    cluster_id = data.get('cluster_id')
                    confidence = data.get('confidence', 0)
                    cluster_name = data.get('cluster_info', {}).get('nome', 'Unknown')
                    
                    self.log_result(
                        "Single Prediction", 
                        True, 
                        f"Cluster {cluster_id} ({cluster_name}), Confiança: {confidence:.3f}"
                    )
                else:
                    self.log_result(
                        "Single Prediction", 
                        False, 
                        "Predição falhou",
                        data
                    )
            else:
                self.log_result(
                    "Single Prediction", 
                    False, 
                    f"Status code: {response.status_code}",
                    response.text
                )
                
        except Exception as e:
            self.log_result("Single Prediction", False, f"Erro: {e}")
    
    def test_batch_prediction(self):
        """Testa predição em lote"""
        
        # Múltiplos candidatos de teste
        candidatos = [
            {
                "codigo_candidato": "BATCH_001",
                "sexo": "Feminino",
                "idade": 25,
                "nivel_academico": "Superior Completo",
                "nivel_ingles": "Intermediário",
                "nivel_profissional": "Júnior",
                "area_atuacao": "Marketing",
                "cv_texto": "Analista de marketing digital com 2 anos de experiência",
                "data_candidatura": "2024-01-10",
                "recrutador": "Carlos Santos",
                "codigo_vaga": "VAGA_002",
                "titulo_vaga": "Analista de Marketing",
                "modalidade": "CLT",
                "cliente": "Marketing Corp",
                "tipo_contratacao": "CLT",
                "cidade_vaga": "Rio de Janeiro",
                "estado_vaga": "RJ",
                "salario_minimo": 4000.0,
                "salario_maximo": 6000.0,
                "experiencia_minima": 1,
                "experiencia_maxima": 3
            },
            {
                "codigo_candidato": "BATCH_002",
                "sexo": "Masculino",
                "idade": 35,
                "nivel_academico": "Pós-graduação",
                "nivel_ingles": "Fluente",
                "nivel_profissional": "Sênior",
                "area_atuacao": "Finanças",
                "cv_texto": "Gerente financeiro com 10 anos de experiência em controladoria",
                "data_candidatura": "2024-01-12",
                "recrutador": "Maria Oliveira",
                "codigo_vaga": "VAGA_003",
                "titulo_vaga": "Gerente Financeiro",
                "modalidade": "CLT",
                "cliente": "Finance Corp",
                "tipo_contratacao": "CLT",
                "cidade_vaga": "Belo Horizonte",
                "estado_vaga": "MG",
                "salario_minimo": 12000.0,
                "salario_maximo": 18000.0,
                "experiencia_minima": 5,
                "experiencia_maxima": 12
            },
            {
                "codigo_candidato": "BATCH_003",
                "sexo": "Feminino",
                "idade": 22,
                "nivel_academico": "Superior Incompleto",
                "nivel_ingles": "Básico",
                "nivel_profissional": "Estagiário",
                "area_atuacao": "Recursos Humanos",
                "cv_texto": "Estudante de psicologia buscando estágio em RH",
                "data_candidatura": "2024-01-08",
                "recrutador": "Pedro Silva",
                "codigo_vaga": "VAGA_004",
                "titulo_vaga": "Estagiário de RH",
                "modalidade": "Estágio",
                "cliente": "HR Corp",
                "tipo_contratacao": "Estágio",
                "cidade_vaga": "Porto Alegre",
                "estado_vaga": "RS",
                "salario_minimo": 1000.0,
                "salario_maximo": 1500.0,
                "experiencia_minima": 0,
                "experiencia_maxima": 1
            }
        ]
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch", 
                json=candidatos,
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list) and len(data) == len(candidatos):
                    successful_predictions = sum(1 for pred in data if pred.get('success', False))
                    
                    self.log_result(
                        "Batch Prediction", 
                        True, 
                        f"{successful_predictions}/{len(candidatos)} predições bem-sucedidas"
                    )
                    
                    # Mostrar resultados detalhados
                    for i, pred in enumerate(data):
                        if pred.get('success', False):
                            cluster_id = pred.get('cluster_id')
                            confidence = pred.get('confidence', 0)
                            print(f"   Candidato {i+1}: Cluster {cluster_id}, Confiança: {confidence:.3f}")
                        else:
                            print(f"   Candidato {i+1}: Erro na predição")
                else:
                    self.log_result(
                        "Batch Prediction", 
                        False, 
                        f"Resposta inválida: esperado lista com {len(candidatos)} itens",
                        data
                    )
            else:
                self.log_result(
                    "Batch Prediction", 
                    False, 
                    f"Status code: {response.status_code}",
                    response.text
                )
                
        except Exception as e:
            self.log_result("Batch Prediction", False, f"Erro: {e}")
    
    def test_edge_cases(self):
        """Testa casos extremos"""
        
        # Candidato com dados mínimos
        candidato_minimo = {
            "codigo_candidato": "EDGE_001",
            "codigo_vaga": "VAGA_EDGE"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict", 
                json=candidato_minimo,
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success', False):
                    self.log_result(
                        "Edge Case - Dados Mínimos", 
                        True, 
                        "Predição com dados mínimos funcionou"
                    )
                else:
                    self.log_result(
                        "Edge Case - Dados Mínimos", 
                        False, 
                        "Predição falhou com dados mínimos",
                        data
                    )
            else:
                self.log_result(
                    "Edge Case - Dados Mínimos", 
                    False, 
                    f"Status code: {response.status_code}",
                    response.text
                )
                
        except Exception as e:
            self.log_result("Edge Case - Dados Mínimos", False, f"Erro: {e}")
    
    def test_performance(self):
        """Testa performance da API"""
        
        candidato_teste = {
            "codigo_candidato": "PERF_001",
            "sexo": "Masculino",
            "idade": 30,
            "nivel_academico": "Superior Completo",
            "nivel_profissional": "Pleno",
            "codigo_vaga": "VAGA_PERF"
        }
        
        # Teste de múltiplas requisições
        num_requests = 10
        times = []
        
        try:
            for i in range(num_requests):
                candidato_teste["codigo_candidato"] = f"PERF_{i:03d}"
                
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict", 
                    json=candidato_teste,
                    timeout=TIMEOUT
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
                else:
                    break
            
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                
                self.log_result(
                    "Performance Test", 
                    True, 
                    f"{len(times)} requisições - Média: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s"
                )
            else:
                self.log_result(
                    "Performance Test", 
                    False, 
                    "Nenhuma requisição bem-sucedida"
                )
                
        except Exception as e:
            self.log_result("Performance Test", False, f"Erro: {e}")
    
    def run_all_tests(self):
        """Executa todos os testes"""
        print("🚀 INICIANDO TESTES DA API - 3 CLUSTERS")
        print("=" * 60)
        
        # Lista de testes
        tests = [
            ("Health Check", self.test_health_check),
            ("Root Endpoint", self.test_root_endpoint),
            ("Clusters Info", self.test_clusters_info),
            ("Model Info", self.test_model_info),
            ("Single Prediction", self.test_single_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Edge Cases", self.test_edge_cases),
            ("Performance", self.test_performance)
        ]
        
        # Executar testes
        for test_name, test_func in tests:
            print(f"\\n🔄 Executando: {test_name}")
            test_func()
            time.sleep(0.5)  # Pequena pausa entre testes
        
        # Resumo dos resultados
        print("\\n" + "=" * 60)
        print("📊 RESUMO DOS TESTES")
        print("=" * 60)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - successful_tests
        
        print(f"✅ Testes bem-sucedidos: {successful_tests}")
        print(f"❌ Testes falharam: {failed_tests}")
        print(f"📊 Taxa de sucesso: {successful_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\\n❌ TESTES QUE FALHARAM:")
            for result in self.results:
                if not result['success']:
                    print(f"   • {result['test_name']}: {result['message']}")
        
        # Salvar resultados
        self.save_results()
        
        return successful_tests == total_tests
    
    def save_results(self):
        """Salva resultados dos testes"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_api_3_clusters_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"\\n💾 Resultados salvos em: {filename}")
            
        except Exception as e:
            print(f"\\n⚠️ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    print("🎯 TESTE DA API DECISION ML - 3 CLUSTERS")
    print("📋 Testando todos os endpoints e funcionalidades")
    print(f"🌐 URL da API: {API_BASE_URL}")
    print("\\n💡 Certifique-se de que a API está rodando:")
    print("   python api_3_clusters_final.py")
    print("\\n" + "=" * 60)
    
    # Aguardar confirmação
    input("Pressione Enter para iniciar os testes...")
    
    # Executar testes
    tester = APITester()
    success = tester.run_all_tests()
    
    if success:
        print("\\n🎉 TODOS OS TESTES PASSARAM!")
        print("✅ API está funcionando corretamente")
    else:
        print("\\n⚠️ ALGUNS TESTES FALHARAM")
        print("🔧 Verifique os logs acima para detalhes")
    
    return success

if __name__ == "__main__":
    main()

