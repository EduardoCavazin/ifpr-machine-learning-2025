import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from random import shuffle
import warnings
import time
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# ===== CONFIGURAÇÕES DO EXPERIMENTO =====
NUM_EMBARALHAMENTOS = 4        # 4 embaralhamentos do dataset completo  
NUM_PARTES_DATASET = 5         # 5 partes de 20% cada (20% teste, 80% treino)
NUM_REPETICOES_POR_PARTE = 1   # 1 repetição para cada parte como teste
# Total: 5 partes × 1 repetição × 4 embaralhamentos = 20 avaliações

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

    # Armazenar probabilidades e labels para ROC
    roc_data = {
        'perceptron': {'y_true': [], 'y_proba': []},
        'svm': {'y_true': [], 'y_proba': []},
        'bayes': {'y_true': [], 'y_proba': []},
        'trees': {'y_true': [], 'y_proba': []},
        'knn': {'y_true': [], 'y_proba': []}
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

                # Capturar probabilidades para ROC (apenas na primeira parte do primeiro embaralhamento)
                if parte_id == 0 and repeticao == 0:
                    if hasattr(classifier, "predict_proba"):
                        y_proba = classifier.predict_proba(xte)
                    elif hasattr(classifier, "decision_function"):
                        y_proba = classifier.decision_function(xte)
                    else:
                        y_proba = ypred  # Fallback para classificadores sem probabilidade

                    roc_data[clf_name]['y_true'].extend(ytrue)
                    roc_data[clf_name]['y_proba'].extend(y_proba)

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
    
    print(f" Resumo das {NUM_PARTES_DATASET} partes executadas.")

    return results_f1, results_accuracy, roc_data


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

    # Armazenar dados de ROC do primeiro embaralhamento
    all_roc_data = None

    warnings.filterwarnings('ignore')

    for embaralhamento_id in range(NUM_EMBARALHAMENTOS):
        print(f"\n--- EMBARALHAMENTO {embaralhamento_id + 1}/{NUM_EMBARALHAMENTOS} ---")

        idx = list(range(len(y)))
        shuffle(idx)
        X_shuffled = X[idx]
        y_shuffled = y[idx]

        fold_results_f1, fold_results_accuracy, roc_data = get_cv_value(X_shuffled, y_shuffled)

        # Salvar dados de ROC apenas do primeiro embaralhamento
        if embaralhamento_id == 0:
            all_roc_data = roc_data

        for clf_name in fold_results_f1.keys():
            for resultado_f1 in fold_results_f1[clf_name]:
                all_results_f1[clf_name].append(resultado_f1)

            for resultado_accuracy in fold_results_accuracy[clf_name]:
                all_results_accuracy[clf_name].append(resultado_accuracy)

        print(f"   Embaralhamento {embaralhamento_id + 1} concluído! (5 execuções adicionadas)")

    warnings.filterwarnings('default')

    return all_results_f1, all_results_accuracy, all_roc_data


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

# ===== SALVAR RESULTADOS EM CSV =====
def salvar_resultados_csv(all_results_f1, all_results_accuracy, estatisticas, roc_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resultados_ml_{timestamp}.csv"

    print(f"\n Salvando resultados em formato horizontal em {filename}...")

    total_execucoes = len(all_results_f1[list(all_results_f1.keys())[0]])
    print(f"    Cada algoritmo tem {total_execucoes} execuções")

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        for clf_name in all_results_f1.keys():
            f1_linha = ['Automobile', clf_name, 'f1_score'] + all_results_f1[clf_name]
            writer.writerow(f1_linha)

            accuracy_linha = ['Automobile', clf_name, 'accuracy'] + all_results_accuracy[clf_name]
            writer.writerow(accuracy_linha)

    print(f"    {filename} salvo em formato horizontal!")
    print(f"    Formato: dataset,algoritmo,metrica,exec1,exec2,...,exec{total_execucoes}")

    # Salvar dados de ROC em arquivo separado
    roc_filename = f"roc_data_{timestamp}.csv"
    print(f"\n Salvando dados ROC em {roc_filename}...")

    with open(roc_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['classifier', 'y_true', 'y_proba'])

        for clf_name in roc_data.keys():
            y_true = roc_data[clf_name]['y_true']
            y_proba = roc_data[clf_name]['y_proba']

            # Salvar cada amostra em uma linha
            for i in range(len(y_true)):
                # Se y_proba for array multidimensional (probabilidades de várias classes)
                if hasattr(y_proba[i], '__len__'):
                    proba_str = '|'.join([str(p) for p in y_proba[i]])
                else:
                    proba_str = str(y_proba[i])

                writer.writerow([clf_name, y_true[i], proba_str])

    print(f"    {roc_filename} salvo com dados para curva ROC!")

    return filename, roc_filename


# ===== VISUALIZAÇÕES COM MATPLOTLIB =====
def criar_visualizacoes(all_results_f1, all_results_accuracy, estatisticas):
    print(f"\n Gerando 2 boxplots (F1-Score e Acurácia)...")
    
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 6)
    plt.rcParams['font.size'] = 11
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    algoritmos_nomes = list(all_results_f1.keys())
    cores = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    
    # ===== BOXPLOT 1: F1-SCORE =====
    dados_f1 = [all_results_f1[nome] for nome in algoritmos_nomes]
    
    box1 = ax1.boxplot(dados_f1, labels=algoritmos_nomes, patch_artist=True)
    
    # Colorir caixas F1-Score
    for patch, cor in zip(box1['boxes'], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.7)
    
    ax1.set_title(' Distribuição F1-Score por Algoritmo', fontsize=13, fontweight='bold', pad=20)
    ax1.set_xlabel('Algoritmos', fontweight='bold')
    ax1.set_ylabel('F1-Score', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # ===== BOXPLOT 2: ACURÁCIA =====
    dados_accuracy = [all_results_accuracy[nome] for nome in algoritmos_nomes]
    
    box2 = ax2.boxplot(dados_accuracy, labels=algoritmos_nomes, patch_artist=True)
    
    # Colorir caixas Acurácia (cores mais escuras)
    cores_accuracy = ['steelblue', 'forestgreen', 'indianred', 'orange', 'mediumvioletred']
    for patch, cor in zip(box2['boxes'], cores_accuracy):
        patch.set_facecolor(cor)
        patch.set_alpha(0.7)
    
    ax2.set_title(' Distribuição Acurácia por Algoritmo', fontsize=13, fontweight='bold', pad=20)
    ax2.set_xlabel('Algoritmos', fontweight='bold')
    ax2.set_ylabel('Acurácia', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar figura
    plt.savefig('boxplots_comparacao.png', dpi=300, bbox_inches='tight')
    print(f"     Boxplots salvos: boxplots_comparacao.png")
    
    # Mostrar gráfico
    plt.show()
    
    print(f"     Visualizações concluídas!")


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
        all_results_f1, all_results_accuracy, roc_data = executar_experimentos(X, y)

        # Calcular e exibir estatísticas
        estatisticas = calcular_estatisticas(all_results_f1)

        # Exibir ranking final
        exibir_ranking(estatisticas)

        # Salvar resultados em CSV
        salvar_resultados_csv(all_results_f1, all_results_accuracy, estatisticas, roc_data)
        
        #  Gerar visualizações (2 boxplots)
        criar_visualizacoes(all_results_f1, all_results_accuracy, estatisticas)
        
        print(f"\n EXPERIMENTO CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
    except Exception as e:
        print(f" ERRO durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
