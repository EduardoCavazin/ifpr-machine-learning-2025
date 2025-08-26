"""
===============================================================================
EXERCÍCIO: COMPARAÇÃO DE ALGORITMOS - DATASET AUTOMOBILE
===============================================================================
Requisitos:
1. Fazer o código funcionar com dataset automobile
2. Dividir dataset em 80% treino / 20% teste
3. Calcular F1-Score para cada algoritmo
4. Acrescentar resultados em listas
5. Repetir 20 vezes para cada algoritmo
6. Calcular média e desvio padrão para cada algoritmo
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from random import shuffle
import warnings
import time


def carregar_dataset_automobile():
    print("Carregando dataset automobile...")
    
    column_names = [
        'symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration',
        'num_of_doors', 'body_style', 'drive_wheels', 'engine_location',
        'wheel_base', 'length', 'width', 'height', 'curb_weight',
        'engine_type', 'num_of_cylinders', 'engine_size', 'fuel_system',
        'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm',
        'city_mpg', 'highway_mpg', 'price'
    ]
    
    df = pd.read_csv('../../Assets/automobile/imports-85.csv', 
                     names=column_names, na_values='?')
    
    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
    return df


def processar_dataset(df):
    print("\n Processando dataset...")
    
    linhas_antes = len(df)
    df_clean = df.dropna()
    linhas_depois = len(df_clean)
    
    print(f"   Dados faltantes removidos: {linhas_antes - linhas_depois} linhas")
    print(f"   Dataset final: {linhas_depois} amostras")
    
    df_processed = df_clean.copy()
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    print(f"   Convertendo {len(categorical_columns)} colunas categóricas...")
    
    le = LabelEncoder()
    for col in categorical_columns:
        df_processed[col] = le.fit_transform(df_processed[col])
    
    print("Dataset processado!")
    return df_processed


def preparar_dados_ml(df, train_ratio=0.8):
    print(f"\nPreparando dados para ML (Treino: {train_ratio*100:.0f}% / Teste: {(1-train_ratio)*100:.0f}%)...")
    
    target_column = 'symboling'
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    indices = list(range(len(y)))
    shuffle(indices)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    split_point = int(len(y) * train_ratio)
    
    X_train = X_shuffled[:split_point]
    y_train = y_shuffled[:split_point]
    X_test = X_shuffled[split_point:]
    y_test = y_shuffled[split_point:]
    
    print(f"Dados preparados: {len(X_train)} treino, {len(X_test)} teste")
    
    return X_train, X_test, y_train, y_test


def executar_experimentos(X_train, X_test, y_train, y_test, num_execucoes=20):
    print(f"\nEXECUTANDO EXPERIMENTOS - {num_execucoes} EXECUÇÕES")
    print("="*70)
    
    print("Normalizando dados...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    algoritmos = {
        'Perceptron': lambda: Perceptron(max_iter=1000, random_state=None),
        'SVM': lambda: SVC(random_state=None),
        'Naive Bayes': lambda: GaussianNB(),
        'Decision Tree': lambda: DecisionTreeClassifier(random_state=None, max_depth=10),
        'KNN': lambda: KNeighborsClassifier(n_neighbors=7)
    }
    
    resultados = {nome: [] for nome in algoritmos.keys()}
    tempos = {nome: [] for nome in algoritmos.keys()}  # Armazenar tempos
    
    warnings.filterwarnings('ignore')
    
    print("\nIniciando execuções...")
    print("Exec | " + " | ".join([f"{nome:>12}" for nome in algoritmos.keys()]) + " | " + " | ".join([f"{nome[:8]}(s)" for nome in algoritmos.keys()]))
    print("-" * (6 + 15 * len(algoritmos) + 12 * len(algoritmos)))
    
    for execucao in range(num_execucoes):
        scores_execucao = []
        tempos_execucao = []
        
        for nome, criar_modelo in algoritmos.items():
            inicio = time.time()
            
            modelo = criar_modelo()
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            
            fim = time.time()
            tempo_execucao = fim - inicio
            
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            resultados[nome].append(f1)
            tempos[nome].append(tempo_execucao)
            
            scores_execucao.append(f1)
            tempos_execucao.append(tempo_execucao)
        
        scores_str = " | ".join([f"{score:>12.4f}" for score in scores_execucao])
        tempos_str = " | ".join([f"{tempo:>9.3f}" for tempo in tempos_execucao])
        print(f"{execucao+1:3d}  | {scores_str} | {tempos_str}")
    
    warnings.filterwarnings('default')
    
    return resultados, tempos


def calcular_estatisticas(resultados, tempos):
    print(f"\nESTATÍSTICAS FINAIS")
    print("="*100)
    
    estatisticas = {}
    
    print(f"{'Algoritmo':<15} | {'F1 Média':<8} | {'F1 Desvio':<8} | {'F1 Min':<8} | {'F1 Max':<8} | {'Tempo Médio':<12} | {'Tempo Total':<12}")
    print("-" * 98)
    
    for nome in resultados.keys():
        scores = resultados[nome]
        times = tempos[nome]
        
        # Estatísticas F1-Score
        media_f1 = np.mean(scores)
        desvio_f1 = np.std(scores)
        minimo_f1 = np.min(scores)
        maximo_f1 = np.max(scores)
        
        # Estatísticas Tempo
        tempo_medio = np.mean(times)
        tempo_total = np.sum(times)
        
        estatisticas[nome] = {
            'media_f1': media_f1,
            'desvio_f1': desvio_f1,
            'minimo_f1': minimo_f1,
            'maximo_f1': maximo_f1,
            'tempo_medio': tempo_medio,
            'tempo_total': tempo_total,
            'scores': scores,
            'tempos': times
        }
        
        print(f"{nome:<15} | {media_f1:<8.4f} | {desvio_f1:<8.4f} | {minimo_f1:<8.4f} | {maximo_f1:<8.4f} | {tempo_medio:<12.3f}s | {tempo_total:<12.3f}s")
    
    return estatisticas


def exibir_ranking(estatisticas):
    print(f"\nRANKING DE PERFORMANCE (por F1-Score médio)")
    print("="*60)
    
    ranking = sorted(estatisticas.items(), key=lambda x: x[1]['media_f1'], reverse=True)
    
    medalhas = ["1º", "2º", "3º", "4º", "5º"]
    
    for i, (nome, stats) in enumerate(ranking):
        medalha = medalhas[i] if i < len(medalhas) else f"{i+1}º"
        print(f"{medalha} {nome:<15}: F1={stats['media_f1']:.4f} (±{stats['desvio_f1']:.4f}) | Tempo={stats['tempo_medio']:.3f}s")
    
    print(f"\nRANKING DE VELOCIDADE (por tempo médio)")
    print("="*60)
    
    ranking_tempo = sorted(estatisticas.items(), key=lambda x: x[1]['tempo_medio'])
    
    for i, (nome, stats) in enumerate(ranking_tempo):
        medalha = medalhas[i] if i < len(medalhas) else f"{i+1}º"
        print(f"{medalha} {nome:<15}: Tempo={stats['tempo_medio']:.3f}s | F1={stats['media_f1']:.4f}")


def main():
    print("EXERCÍCIO: COMPARAÇÃO DE ALGORITMOS ML")
    print("="*70)
    
    try:
        df = carregar_dataset_automobile()
        
        df_processed = processar_dataset(df)
        
        X_train, X_test, y_train, y_test = preparar_dados_ml(df_processed, train_ratio=0.8)
        
        resultados, tempos = executar_experimentos(X_train, X_test, y_train, y_test, num_execucoes=20)
        
        estatisticas = calcular_estatisticas(resultados, tempos)
        
        exibir_ranking(estatisticas)
        
        print(f"\nEXPERIMENTO CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
    except Exception as e:
        print(f"ERRO durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

