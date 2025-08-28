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
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def preparar_dados_ml(df):
    print(f"\nPreparando dados para Cross Validation...")
    
    target_column = 'symboling'
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    print(f"Dados preparados: {len(X)} amostras para cross validation")
    
    return X, y


def executar_experimentos(X, y, num_ciclos=3, num_execucoes=20, k_folds=5):
    print(f"\nEXECUTANDO {num_ciclos} CICLOS DE CROSS VALIDATION")
    print(f"Cada fold: {num_execucoes} execuções independentes")
    print(f"Total por ciclo: {k_folds} folds × {num_execucoes} execuções = {k_folds * num_execucoes} avaliações")
    print(f"Total geral: {num_ciclos * k_folds * num_execucoes} avaliações por algoritmo")
    print("="*80)
    
    print("Normalizando dados...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    algoritmos = {
        'Perceptron': lambda: Perceptron(max_iter=1000, random_state=None),
        'SVM': lambda: SVC(random_state=None),
        'Naive Bayes': lambda: GaussianNB(),
        'Decision Tree': lambda: DecisionTreeClassifier(random_state=None, max_depth=10),
        'KNN': lambda: KNeighborsClassifier(n_neighbors=7)
    }
    
    # Armazenar TODOS os scores individuais
    todos_resultados = {nome: [] for nome in algoritmos.keys()}
    todos_tempos = {nome: [] for nome in algoritmos.keys()}
    
    # Resultados por ciclo (média dos 5 folds)
    resultados_por_ciclo = []
    
    warnings.filterwarnings('ignore')
    
    for ciclo in range(num_ciclos):
        print(f"\n--- CICLO {ciclo + 1} ---")
        
        medias_folds_ciclo = {nome: [] for nome in algoritmos.keys()}
        
        # Criar os folds uma vez para este ciclo
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=ciclo)
        folds = list(skf.split(X_scaled, y))
        
        # Para cada fold
        for fold_idx in range(k_folds):
            print(f"\n  -- FOLD {fold_idx + 1} --")
            print("Exec | " + " | ".join([f"{nome:>12}" for nome in algoritmos.keys()]) + " | " + " | ".join([f"{nome[:8]}(s)" for nome in algoritmos.keys()]))
            print("-" * (6 + 15 * len(algoritmos) + 12 * len(algoritmos)))
            
            train_idx, test_idx = folds[fold_idx]
            X_train_fold = X_scaled[train_idx]
            X_test_fold = X_scaled[test_idx]
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]
            
            scores_fold = {nome: [] for nome in algoritmos.keys()}
            
            # 20 execuções para este fold
            for execucao in range(num_execucoes):
                scores_execucao = []
                tempos_execucao = []
                
                for nome, criar_modelo in algoritmos.items():
                    inicio = time.time()
                    
                    modelo = criar_modelo()
                    modelo.fit(X_train_fold, y_train_fold)
                    y_pred = modelo.predict(X_test_fold)
                    f1 = f1_score(y_test_fold, y_pred, average='macro', zero_division=0)
                    
                    fim = time.time()
                    tempo_execucao = fim - inicio
                    
                    # Armazenar score e tempo
                    scores_fold[nome].append(f1)
                    todos_resultados[nome].append(f1)
                    todos_tempos[nome].append(tempo_execucao)
                    
                    scores_execucao.append(f1)
                    tempos_execucao.append(tempo_execucao)
                
                scores_str = " | ".join([f"{score:>12.4f}" for score in scores_execucao])
                tempos_str = " | ".join([f"{tempo:>9.3f}" for tempo in tempos_execucao])
                print(f"{execucao+1:3d}  | {scores_str} | {tempos_str}")
            
            # Calcular média do fold (20 execuções)
            for nome in algoritmos.keys():
                media_fold = np.mean(scores_fold[nome])
                medias_folds_ciclo[nome].append(media_fold)
            
            medias_fold_str = " | ".join([f"{np.mean(scores_fold[nome]):>12.4f}" for nome in algoritmos.keys()])
            print(f"Média Fold {fold_idx + 1}: {medias_fold_str}")
        
        # Calcular média do ciclo (média dos 5 folds)
        medias_ciclo = {}
        for nome in algoritmos.keys():
            medias_ciclo[nome] = np.mean(medias_folds_ciclo[nome])
        
        resultados_por_ciclo.append(medias_ciclo)
        
        print(f"\nMédia Ciclo {ciclo + 1}: " + " | ".join([f"{medias_ciclo[nome]:>12.4f}" for nome in algoritmos.keys()]))
    
    warnings.filterwarnings('default')
    
    return todos_resultados, todos_tempos, resultados_por_ciclo


def calcular_estatisticas(resultados, tempos, resultados_por_ciclo):
    print(f"\nESTATÍSTICAS DOS 3 CICLOS DE CROSS VALIDATION")
    print("="*120)
    
    # Mostrar médias por ciclo (cada ciclo tem 100 scores: 20 execuções × 5 folds)
    print(f"{'Algoritmo':<15} | {'Ciclo 1':<10} | {'Ciclo 2':<10} | {'Ciclo 3':<10} | {'Média Final':<12} | {'Desvio Ciclos':<12}")
    print("-" * 85)
    
    algoritmos = list(resultados.keys())
    medias_finais = {}
    
    for nome in algoritmos:
        ciclo1 = resultados_por_ciclo[0][nome]  # Média de 100 scores
        ciclo2 = resultados_por_ciclo[1][nome]  # Média de 100 scores  
        ciclo3 = resultados_por_ciclo[2][nome]  # Média de 100 scores
        
        media_final = (ciclo1 + ciclo2 + ciclo3) / 3
        desvio_ciclos = np.std([ciclo1, ciclo2, ciclo3])
        
        medias_finais[nome] = media_final
        
        print(f"{nome:<15} | {ciclo1:<10.4f} | {ciclo2:<10.4f} | {ciclo3:<10.4f} | {media_final:<12.4f} | {desvio_ciclos:<12.4f}")
    
    print(f"\nESTATÍSTICAS DETALHADAS - TODOS OS 300 SCORES")
    print("="*120)
    
    estatisticas = {}
    
    print(f"{'Algoritmo':<15} | {'F1 Média (300)':<12} | {'F1 Desvio (300)':<12} | {'F1 Min':<10} | {'F1 Max':<10} | {'Tempo Médio':<12} | {'Tempo Total':<12}")
    print("-" * 118)
    
    for nome in algoritmos:
        scores = resultados[nome]  # 300 scores individuais
        times = tempos[nome]      # 60 tempos (20 execuções × 3 ciclos)
        
        # Estatísticas F1-Score de todos os 300 scores individuais
        media_f1 = np.mean(scores)
        desvio_f1 = np.std(scores)
        minimo_f1 = np.min(scores)
        maximo_f1 = np.max(scores)
        
        # Estatísticas Tempo (60 execuções)
        tempo_medio = np.mean(times)
        tempo_total = np.sum(times)
        
        estatisticas[nome] = {
            'media_f1': media_f1,
            'desvio_f1': desvio_f1,
            'minimo_f1': minimo_f1,
            'maximo_f1': maximo_f1,
            'tempo_medio': tempo_medio,
            'tempo_total': tempo_total,
            'media_final_ciclos': medias_finais[nome],
            'scores': scores,
            'tempos': times,
            'total_scores': len(scores),
            'total_execucoes': len(times)
        }
        
        print(f"{nome:<15} | {media_f1:<12.4f} | {desvio_f1:<12.4f} | {minimo_f1:<10.4f} | {maximo_f1:<10.4f} | {tempo_medio:<12.3f}s | {tempo_total:<12.3f}s")
    
    print(f"\nResumo: {len(scores)} scores individuais por algoritmo ({3} ciclos × {20} execuções × {5} folds)")
    
    return estatisticas


def exibir_ranking(estatisticas):
    print(f"\nRANKING BASEADO NA MÉDIA DOS 3 CICLOS")
    print("="*70)
    
    ranking = sorted(estatisticas.items(), key=lambda x: x[1]['media_final_ciclos'], reverse=True)
    
    medalhas = ["1º", "2º", "3º", "4º", "5º"]
    
    for i, (nome, stats) in enumerate(ranking):
        medalha = medalhas[i] if i < len(medalhas) else f"{i+1}º"
        print(f"{medalha} {nome:<15}: Média Final={stats['media_final_ciclos']:.4f} | F1 Geral={stats['media_f1']:.4f} (±{stats['desvio_f1']:.4f})")
    
    print(f"\nRANKING DE VELOCIDADE (por tempo médio)")
    print("="*70)
    
    ranking_tempo = sorted(estatisticas.items(), key=lambda x: x[1]['tempo_medio'])
    
    for i, (nome, stats) in enumerate(ranking_tempo):
        medalha = medalhas[i] if i < len(medalhas) else f"{i+1}º"
        print(f"{medalha} {nome:<15}: Tempo={stats['tempo_medio']:.3f}s | Média Final={stats['media_final_ciclos']:.4f}")


def main():
    print("EXERCÍCIO: COMPARAÇÃO DE ALGORITMOS ML")
    print("="*70)
    
    try:
        df = carregar_dataset_automobile()
        
        df_processed = processar_dataset(df)
        
        X, y = preparar_dados_ml(df_processed)
        
        resultados, tempos, resultados_por_ciclo = executar_experimentos(X, y, num_ciclos=3, num_execucoes=20, k_folds=5)
        
        estatisticas = calcular_estatisticas(resultados, tempos, resultados_por_ciclo)
        
        exibir_ranking(estatisticas)
        
        print(f"\nEXPERIMENTO CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
    except Exception as e:
        print(f"ERRO durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
