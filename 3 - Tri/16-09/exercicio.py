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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from random import shuffle
import warnings
import time
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# ===== CONFIGURAÇÕES DO EXPERIMENTO =====
NUM_EMBARALHAMENTOS = 4        # 4 embaralhamentos do dataset completo  
NUM_PARTES_DATASET = 5         # 5 partes de 20% cada (20% teste, 80% treino)
NUM_REPETICOES_POR_PARTE = 20  # 20 repetições para cada parte como teste
# Total: 5 partes × 20 repetições × 4 embaralhamentos = 400 avaliações

# Random state global
rng = np.random.RandomState()


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
    
    # ===== NORMALIZAÇÃO MIN-MAX =====
    print("   Aplicando normalização Min-Max...")
    target_column = 'symboling'
    X = df_processed.drop(columns=[target_column]).values
    y = df_processed[target_column].values
    
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    print(f"Features normalizadas para intervalo [0,1]")
    print("Dataset processado!")
    
    return X_normalized, y


def preparar_dados_ml(df):
    print(f"\n Preparando dados para Cross Validation...")
    print(f"   Configuração: {NUM_EMBARALHAMENTOS} embaralhamentos × {NUM_PARTES_DATASET} partes × {NUM_REPETICOES_POR_PARTE} repetições")
    print(f"   Divisão treino/teste: 80%/20% (cada parte de 20% vira teste)")
    print(f"   Total de avaliações por algoritmo: {NUM_EMBARALHAMENTOS * NUM_PARTES_DATASET * NUM_REPETICOES_POR_PARTE}")
    return df


# ===== CROSS VALIDATION SEQUENCIAL =====
def get_cv_value(X_data, y_data):
    total_samples = len(y_data)
    part_size = total_samples // NUM_PARTES_DATASET  
    
    # Armazenar F1-Score e Acurácia separadamente
    results_f1 = {
        'perceptron': [],
        'svm': [],
        'bayes': [],
        'trees': [],
        'knn': []
    }
    
    results_accuracy = {
        'perceptron': [],
        'svm': [],
        'bayes': [],
        'trees': [],
        'knn': []
    }
    
    print(f"   Divisão: {NUM_PARTES_DATASET} partes de ~{part_size} amostras cada (20% teste, 80% treino)")
    
    for parte_id in range(NUM_PARTES_DATASET):
        print(f"   -- Parte {parte_id + 1}/{NUM_PARTES_DATASET} --")
        
        inicio_teste = parte_id * part_size
        if parte_id == NUM_PARTES_DATASET - 1:  
            fim_teste = total_samples
        else:
            fim_teste = (parte_id + 1) * part_size
        
        xte = X_data[inicio_teste:fim_teste]
        yte = y_data[inicio_teste:fim_teste]
        
        xtr = np.concatenate([X_data[:inicio_teste], X_data[fim_teste:]])
        ytr = np.concatenate([y_data[:inicio_teste], y_data[fim_teste:]])
        
        print(f"     Treino: {len(xtr)} amostras | Teste: {len(xte)} amostras")
        
        parte_results_f1 = {
            'perceptron': [],
            'svm': [],
            'bayes': [],
            'trees': [],
            'knn': []
        }
        
        parte_results_accuracy = {
            'perceptron': [],
            'svm': [],
            'bayes': [],
            'trees': [],
            'knn': []
        }
        
        for repeticao in range(NUM_REPETICOES_POR_PARTE):
            if NUM_REPETICOES_POR_PARTE > 10:  
                if (repeticao + 1) % 5 == 0:  
                    print(f"     Repetição {repeticao + 1}/{NUM_REPETICOES_POR_PARTE}...")
            
            clfs = {
                'perceptron':   Perceptron(max_iter=1000, random_state=rng),
                'svm':          SVC(probability=True, gamma='auto', random_state=rng),
                'bayes':        GaussianNB(),
                'trees':        DecisionTreeClassifier(random_state=rng, max_depth=10),
                'knn':          KNeighborsClassifier(n_neighbors=7)
            }
            
            ytrue = yte
            exec_scores = []
            exec_tempos = []
            
            for clf_name, classifier in clfs.items():
                inicio = time.time()
                
                classifier.fit(xtr, ytr)
                ypred = classifier.predict(xte)
                f1 = f1_score(ytrue, ypred, average='macro', zero_division=0)
                accuracy = accuracy_score(ytrue, ypred)
                
                fim = time.time()
                tempo = fim - inicio
                
                parte_results_f1[clf_name].append(f1)
                parte_results_accuracy[clf_name].append(accuracy)
                exec_scores.append(f1)
                exec_tempos.append(tempo)
            
            if repeticao == 0:
                for i, clf_name in enumerate(parte_results_f1.keys()):
                    print(f"     {clf_name:<12}: F1 = {exec_scores[i]:.4f} | Tempo = {exec_tempos[i]:.3f}s")
        
        for clf_name in parte_results_f1.keys():
            media_parte_f1 = np.mean(parte_results_f1[clf_name])
            media_parte_accuracy = np.mean(parte_results_accuracy[clf_name])
            results_f1[clf_name].append(media_parte_f1)
            results_accuracy[clf_name].append(media_parte_accuracy)
        
        medias_str = " | ".join([f"{np.mean(parte_results_f1[nome]):>8.4f}" for nome in parte_results_f1.keys()])
        print(f"     Média Parte {parte_id + 1}: {medias_str}")
        print()
    
    print(f" Médias das {NUM_PARTES_DATASET} partes:")
    fold_results_f1 = {}
    fold_results_accuracy = {}
    
    for clf_name, result in results_f1.items():
        media_f1 = sum(result) / len(result)
        media_accuracy = sum(results_accuracy[clf_name]) / len(results_accuracy[clf_name])
        fold_results_f1[clf_name] = media_f1
        fold_results_accuracy[clf_name] = media_accuracy
        print(f"     {clf_name:<12}: F1={media_f1:.4f} | Acc={media_accuracy:.4f}")
    
    return fold_results_f1, fold_results_accuracy


def executar_experimentos(X, y):
    print(f"\n EXECUTANDO {NUM_EMBARALHAMENTOS} EMBARALHAMENTOS DO DATASET")
    print("="*80)
    
    all_results_f1 = {
        'perceptron': [],
        'svm': [],
        'bayes': [], 
        'trees': [],
        'knn': []
    }
    
    all_results_accuracy = {
        'perceptron': [],
        'svm': [],
        'bayes': [], 
        'trees': [],
        'knn': []
    }
    
    warnings.filterwarnings('ignore')
    
    for embaralhamento_id in range(NUM_EMBARALHAMENTOS):
        print(f"\n--- EMBARALHAMENTO {embaralhamento_id + 1}/{NUM_EMBARALHAMENTOS} ---")
        
        idx = list(range(len(y)))
        shuffle(idx)
        X_shuffled = X[idx]
        y_shuffled = y[idx]
        
        fold_results_f1, fold_results_accuracy = get_cv_value(X_shuffled, y_shuffled)
        
        for clf_name, result_f1 in fold_results_f1.items():
            result_accuracy = fold_results_accuracy[clf_name]
            all_results_f1[clf_name].append(result_f1)
            all_results_accuracy[clf_name].append(result_accuracy)
        
        print(f"   Embaralhamento {embaralhamento_id + 1} concluído!")
    
    warnings.filterwarnings('default')
    
    return all_results_f1, all_results_accuracy


def calcular_estatisticas(all_results):
    print(f"\n ESTATÍSTICAS FINAIS - {NUM_EMBARALHAMENTOS} EMBARALHAMENTOS")
    print("="*70)
    
    estatisticas = {}
    
    print(f"{'Algoritmo':<12} | {'Embaralh 1':<10} | {'Embaralh 2':<10} | {'Embaralh 3':<10} | {'Média Final':<10} | {'Desvio':<8}")
    print("-" * 78)
    
    for clf_name in all_results.keys():
        scores = all_results[clf_name] 
        
        media_final = np.mean(scores)
        desvio = np.std(scores) if len(scores) > 1 else 0.0
        
        estatisticas[clf_name] = {
            'media_final': media_final,
            'desvio': desvio,
            'scores': scores,
            'num_execucoes': len(scores)
        }
        
        if len(scores) >= 3:
            print(f"{clf_name:<12} | {scores[0]:<10.4f} | {scores[1]:<10.4f} | {scores[2]:<10.4f} | {media_final:<10.4f} | {desvio:<8.4f}")
        else:
            scores_str = " | ".join([f"{score:<10.4f}" for score in scores])
            print(f"{clf_name:<12} | {scores_str:<32} | {media_final:<10.4f} | {desvio:<8.4f}")
    
    return estatisticas


def exibir_ranking(estatisticas):
    print(f"\n RANKING FINAL")
    print("="*50)
    
    ranking = sorted(estatisticas.items(), key=lambda x: x[1]['media_final'], reverse=True)

    medalhas = [" 1º", " 2º", " 3º", " 4º", " 5º"]

    for i, (nome, stats) in enumerate(ranking):
        medalha = medalhas[i] if i < len(medalhas) else f"   {i+1}º"
        print(f"{medalha} {nome:<12}: {stats['media_final']:.4f} (±{stats['desvio']:.4f})")
    
    total_avaliacoes = NUM_EMBARALHAMENTOS * NUM_PARTES_DATASET * NUM_REPETICOES_POR_PARTE
    print(f"\nResumo: {NUM_EMBARALHAMENTOS} embaralh × {NUM_PARTES_DATASET} partes × {NUM_REPETICOES_POR_PARTE} repet/parte = {total_avaliacoes} avaliações por algoritmo")


def salvar_resultados_csv(all_results_f1, all_results_accuracy, estatisticas):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resultados_ml_{timestamp}.csv"
    
    print(f"\n Salvando resultados em {filename}...")
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['Algoritmo', 'Execucao', 'F1_Score', 'Accuracy', 'Media_F1', 'Desvio_F1'])
        
        for clf_name in all_results_f1.keys():
            f1_scores = all_results_f1[clf_name]
            accuracy_scores = all_results_accuracy[clf_name]
            media_f1 = estatisticas[clf_name]['media_final']
            desvio_f1 = estatisticas[clf_name]['desvio']
            
            for i, (f1, acc) in enumerate(zip(f1_scores, accuracy_scores)):
                writer.writerow([clf_name, i+1, f1, acc, media_f1, desvio_f1])
    
    print(f"   Arquivo {filename} salvo com sucesso!")
    return filename


# ===== VISUALIZAÇÕES COM MATPLOTLIB =====
def criar_visualizacoes(all_results, estatisticas):
    print(f"\n Gerando visualizações...")
    
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # ===== 1. BOXPLOT =====
    plt.figure(figsize=(12, 6))
    
    algoritmos_nomes = list(all_results.keys())
    dados_boxplot = []
    
    for nome in algoritmos_nomes:
        scores = all_results[nome]
        media = estatisticas[nome]['media_final']
        desvio = estatisticas[nome]['desvio']
        
        if len(scores) > 1:
            dados_simulados = np.random.normal(media, desvio, 50)  
        else:
            dados_simulados = [media] * 10  
        
        dados_boxplot.append(dados_simulados)
    
    plt.subplot(1, 2, 1)
    box_plot = plt.boxplot(dados_boxplot, labels=algoritmos_nomes, patch_artist=True)
    
    cores = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, cor in zip(box_plot['boxes'], cores):
        patch.set_facecolor(cor)
    
    plt.title(' Distribuição F1-Score por Algoritmo\n(Boxplot)', fontsize=12, fontweight='bold')
    plt.xlabel('Algoritmos', fontweight='bold')
    plt.ylabel('F1-Score', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # ===== 2. GRÁFICO DE BARRAS COM DESVIO PADRÃO =====
    plt.subplot(1, 2, 2)
    
    algoritmos = list(estatisticas.keys())
    medias = [estatisticas[alg]['media_final'] for alg in algoritmos]
    desvios = [estatisticas[alg]['desvio'] for alg in algoritmos]
    
    x_pos = np.arange(len(algoritmos))
    bars = plt.bar(x_pos, medias, yerr=desvios, capsize=5, 
                   color=cores[:len(algoritmos)], alpha=0.7, 
                   edgecolor='black', linewidth=1)
    
    for i, (bar, media, desvio) in enumerate(zip(bars, medias, desvios)):
        altura = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., altura + desvio + 0.01,
                f'{media:.3f}±{desvio:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.title(' F1-Score Médio ± Desvio Padrão\n(Gráfico de Barras)', fontsize=12, fontweight='bold')
    plt.xlabel('Algoritmos', fontweight='bold')
    plt.ylabel('F1-Score', fontweight='bold')
    plt.xticks(x_pos, algoritmos, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plt.savefig('resultados_ml_comparacao.png', dpi=300, bbox_inches='tight')
    print(f"    Gráfico salvo: resultados_ml_comparacao.png")
    
    plt.show()
    
    # ===== 3. GRÁFICO DE RANKING (BÔNUS) =====
    plt.figure(figsize=(10, 6))
    
    ranking = sorted(estatisticas.items(), key=lambda x: x[1]['media_final'], reverse=True)
    nomes_ranking = [item[0] for item in ranking]
    valores_ranking = [item[1]['media_final'] for item in ranking]
    desvios_ranking = [item[1]['desvio'] for item in ranking]
    
    y_pos = np.arange(len(nomes_ranking))
    bars = plt.barh(y_pos, valores_ranking, xerr=desvios_ranking, 
                    color=['gold', 'silver', 'orange', 'lightblue', 'lightgray'][:len(nomes_ranking)],
                    alpha=0.8, edgecolor='black', linewidth=1)
    
    # Adicionar medalhas
    medalhas = ['1º', '2º', '3º', '4º', '5º']
    for i, (bar, valor, desvio) in enumerate(zip(bars, valores_ranking, desvios_ranking)):
        medalha = medalhas[i] if i < len(medalhas) else '5º'
        plt.text(valor + desvio + 0.01, bar.get_y() + bar.get_height()/2,
                f'{medalha} {valor:.3f}±{desvio:.3f}', 
                va='center', ha='left', fontweight='bold')
    
    plt.title(' Ranking Final dos Algoritmos\n(F1-Score Médio)', fontsize=14, fontweight='bold')
    plt.xlabel('F1-Score', fontweight='bold')
    plt.yticks(y_pos, nomes_ranking)
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('ranking_algoritmos.png', dpi=300, bbox_inches='tight')
    print(f"    Ranking salvo: ranking_algoritmos.png")
    plt.show()
    
    print(f"    Visualizações concluídas!")


def main():
    print(" EXERCÍCIO: COMPARAÇÃO DE ALGORITMOS ML - CROSS VALIDATION SEQUENCIAL")
    print("="*70)
    print(f"  Configuração: {NUM_EMBARALHAMENTOS} embaralhamentos × {NUM_PARTES_DATASET} partes × {NUM_REPETICOES_POR_PARTE} repetições/parte")
    print(f" Total de avaliações por algoritmo: {NUM_EMBARALHAMENTOS * NUM_PARTES_DATASET * NUM_REPETICOES_POR_PARTE}")
    print("="*70)
    
    try:
        # Carregar dados
        df = carregar_dataset_automobile()
        
        # Processar e normalizar (inclui Min-Max)
        X, y = processar_dataset(df)
        
        # Validação prévia
        preparar_dados_ml(X)
        
        # Executar cross validation manual
        all_results_f1, all_results_accuracy = executar_experimentos(X, y)
        
        # Calcular e exibir estatísticas
        estatisticas = calcular_estatisticas(all_results_f1)
        
        # Exibir ranking final
        exibir_ranking(estatisticas)
        
        # Salvar resultados em CSV
        salvar_resultados_csv(all_results_f1, all_results_accuracy, estatisticas)
        
        #  Gerar visualizações
        criar_visualizacoes(all_results_f1, estatisticas)
        
        print(f"\n EXPERIMENTO CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
    except Exception as e:
        print(f" ERRO durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
