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
from sklearn.metrics import f1_score
from random import shuffle
import warnings
import time

# ===== CONFIGURAÇÕES DO EXPERIMENTO =====
NUM_EXECUCOES_PRINCIPAIS = 3    # Quantas execuções principais (como o professor)
NUM_FOLDS_CV = 5               # Quantos folds no cross validation
NUM_EXECUCOES_POR_FOLD = 20     # Quantas execuções por fold (1 = estilo professor, 20 = estilo original)
TRAIN_RATIO = 0.8              # Proporção treino/teste (80%/20%)

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
    print("\n🔄 Processando dataset...")
    
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
    
    print(f"   Features normalizadas para intervalo [0,1]")
    print("Dataset processado!")
    
    return X_normalized, y


def preparar_dados_ml(df):
    print(f"\n🎯 Preparando dados para Cross Validation...")
    print(f"   Configuração: {NUM_EXECUCOES_PRINCIPAIS} execuções × {NUM_FOLDS_CV} folds × {NUM_EXECUCOES_POR_FOLD} exec/fold")
    print(f"   Divisão treino/teste: {TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}%")
    print(f"   Total de avaliações por algoritmo: {NUM_EXECUCOES_PRINCIPAIS * NUM_FOLDS_CV * NUM_EXECUCOES_POR_FOLD}")
    return df


# ===== CROSS VALIDATION MANUAL (ESTILO PROFESSOR) =====
def get_cv_value(X_data, y_data):
    """
    Implementa cross validation manual com rotação de dados
    Baseado no código do professor + flexibilidade de execuções por fold
    """
    part = int(len(y_data) * TRAIN_RATIO)  # 80% para treino
    
    results = {
        'perceptron': [],
        'svm': [],
        'bayes': [],
        'trees': [],
        'knn': []
    }
    
    print(f"   Divisão por fold: {part} treino / {len(y_data) - part} teste")
    
    for crossv in range(NUM_FOLDS_CV):
        print(f"   -- Fold {crossv + 1}/{NUM_FOLDS_CV} --")
        
        # Dividir dados para este fold
        xtr = X_data[:part]  # x_treino
        ytr = y_data[:part]  # y_treino  
        xte = X_data[part:]  # x_teste
        yte = y_data[part:]  # y_teste
        
        # Resultados para este fold específico
        fold_results = {
            'perceptron': [],
            'svm': [],
            'bayes': [],
            'trees': [],
            'knn': []
        }
        
        # Executar N vezes este fold (flexibilidade nova!)
        for exec_fold in range(NUM_EXECUCOES_POR_FOLD):
            if NUM_EXECUCOES_POR_FOLD > 1:
                print(f"     Exec {exec_fold + 1}/{NUM_EXECUCOES_POR_FOLD}:")
            
            # Criar modelos (igual ao professor)
            clfs = {
                'perceptron':   Perceptron(max_iter=1000, random_state=rng),
                'svm':          SVC(probability=True, gamma='auto', random_state=rng),
                'bayes':        GaussianNB(),
                'trees':        DecisionTreeClassifier(random_state=rng, max_depth=10),
                'knn':          KNeighborsClassifier(n_neighbors=7)
            }
            
            # Treinar e avaliar cada classificador
            ytrue = yte
            exec_scores = []
            exec_tempos = []
            
            for clf_name, classifier in clfs.items():
                inicio = time.time()
                
                classifier.fit(xtr, ytr)
                ypred = classifier.predict(xte)
                f1 = f1_score(ytrue, ypred, average='macro', zero_division=0)
                
                fim = time.time()
                tempo = fim - inicio
                
                fold_results[clf_name].append(f1)
                exec_scores.append(f1)
                exec_tempos.append(tempo)
            
            # Exibir resultados desta execução específica
            if NUM_EXECUCOES_POR_FOLD == 1:
                # Estilo professor (1 linha por fold)
                for i, clf_name in enumerate(fold_results.keys()):
                    print(f"     {clf_name:<12}: F1 = {exec_scores[i]:.4f} | Tempo = {exec_tempos[i]:.3f}s")
            else:
                # Estilo tabela (múltiplas execuções)
                scores_str = " | ".join([f"{score:>8.4f}" for score in exec_scores])
                tempos_str = " | ".join([f"{tempo:>6.3f}s" for tempo in exec_tempos])
                print(f"     {exec_fold+1:2d}: {scores_str} | {tempos_str}")
        
        # Calcular média do fold (todas as execuções deste fold)
        for clf_name in fold_results.keys():
            media_fold = np.mean(fold_results[clf_name])
            results[clf_name].append(media_fold)
        
        if NUM_EXECUCOES_POR_FOLD > 1:
            medias_str = " | ".join([f"{np.mean(fold_results[nome]):>8.4f}" for nome in fold_results.keys()])
            print(f"     Média Fold {crossv + 1}: {medias_str}")
        
        # Rotacionar dados para próximo fold (como o professor)
        y_data = list(y_data[part:]) + list(y_data[:part])
        X_data = list(X_data[part:]) + list(X_data[:part])
        
        print()
    
    # Calcular médias dos 5 folds
    print(f"   📊 Médias dos {NUM_FOLDS_CV} folds:")
    fold_results = {}
    for clf_name, result in results.items():
        media = sum(result) / len(result)
        fold_results[clf_name] = media
        print(f"     {clf_name:<12}: {media:.4f}")
    
    return fold_results


def executar_experimentos(X, y):
    """
    Executa múltiplas rodadas de cross validation
    Baseado na estrutura do professor
    """
    print(f"\n🚀 EXECUTANDO {NUM_EXECUCOES_PRINCIPAIS} EXECUÇÕES DE CROSS VALIDATION")
    print("="*80)
    
    # Armazenar resultados de todas as execuções
    all_results = {
        'perceptron': [],
        'svm': [],
        'bayes': [], 
        'trees': [],
        'knn': []
    }
    
    warnings.filterwarnings('ignore')
    
    for exec_id in range(NUM_EXECUCOES_PRINCIPAIS):
        print(f"\n--- EXECUÇÃO {exec_id + 1}/{NUM_EXECUCOES_PRINCIPAIS} ---")
        
        # Embaralhar dados (como o professor)
        idx = list(range(len(y)))
        shuffle(idx)
        X_shuffled = X[idx]
        y_shuffled = y[idx]
        
        # Executar cross validation para estes dados embaralhados
        fold_results = get_cv_value(X_shuffled, y_shuffled)
        
        # Armazenar resultados desta execução
        for clf_name, result in fold_results.items():
            all_results[clf_name].append(result)
        
        print(f"   ✅ Execução {exec_id + 1} concluída!")
    
    warnings.filterwarnings('default')
    
    return all_results


def calcular_estatisticas(all_results):
    """
    Calcula estatísticas finais baseado no padrão do professor
    """
    print(f"\n📊 ESTATÍSTICAS FINAIS - {NUM_EXECUCOES_PRINCIPAIS} EXECUÇÕES")
    print("="*70)
    
    estatisticas = {}
    
    print(f"{'Algoritmo':<12} | {'Exec 1':<8} | {'Exec 2':<8} | {'Exec 3':<8} | {'Média Final':<10} | {'Desvio':<8}")
    print("-" * 68)
    
    for clf_name in all_results.keys():
        scores = all_results[clf_name]  # Lista com N resultados (1 por execução)
        
        # Estatísticas
        media_final = np.mean(scores)
        desvio = np.std(scores) if len(scores) > 1 else 0.0
        
        estatisticas[clf_name] = {
            'media_final': media_final,
            'desvio': desvio,
            'scores': scores,
            'num_execucoes': len(scores)
        }
        
        # Mostrar resultados individuais se tiver exatamente 3 execuções
        if len(scores) >= 3:
            print(f"{clf_name:<12} | {scores[0]:<8.4f} | {scores[1]:<8.4f} | {scores[2]:<8.4f} | {media_final:<10.4f} | {desvio:<8.4f}")
        else:
            scores_str = " | ".join([f"{score:<8.4f}" for score in scores])
            print(f"{clf_name:<12} | {scores_str:<26} | {media_final:<10.4f} | {desvio:<8.4f}")
    
    return estatisticas


def exibir_ranking(estatisticas):
    """
    Exibe ranking final dos algoritmos
    """
    print(f"\n🏆 RANKING FINAL")
    print("="*50)
    
    # Ordenar por média final (decrescente)
    ranking = sorted(estatisticas.items(), key=lambda x: x[1]['media_final'], reverse=True)
    
    medalhas = ["🥇 1º", "🥈 2º", "🥉 3º", "   4º", "   5º"]
    
    for i, (nome, stats) in enumerate(ranking):
        medalha = medalhas[i] if i < len(medalhas) else f"   {i+1}º"
        print(f"{medalha} {nome:<12}: {stats['media_final']:.4f} (±{stats['desvio']:.4f})")
    
    total_avaliacoes = NUM_EXECUCOES_PRINCIPAIS * NUM_FOLDS_CV * NUM_EXECUCOES_POR_FOLD
    print(f"\nResumo: {NUM_EXECUCOES_PRINCIPAIS} exec × {NUM_FOLDS_CV} folds × {NUM_EXECUCOES_POR_FOLD} exec/fold = {total_avaliacoes} avaliações por algoritmo")


def main():
    print("🚀 EXERCÍCIO: COMPARAÇÃO DE ALGORITMOS ML - ESTILO PROFESSOR")
    print("="*70)
    print(f"⚙️  Configuração: {NUM_EXECUCOES_PRINCIPAIS} exec × {NUM_FOLDS_CV} folds × {NUM_EXECUCOES_POR_FOLD} exec/fold × {TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}% split")
    print(f"📊 Total de avaliações por algoritmo: {NUM_EXECUCOES_PRINCIPAIS * NUM_FOLDS_CV * NUM_EXECUCOES_POR_FOLD}")
    print("="*70)
    
    try:
        # Carregar dados
        df = carregar_dataset_automobile()
        
        # Processar e normalizar (inclui Min-Max)
        X, y = processar_dataset(df)
        
        # Validação prévia
        preparar_dados_ml(X)
        
        # Executar cross validation manual
        all_results = executar_experimentos(X, y)
        
        # Calcular e exibir estatísticas
        estatisticas = calcular_estatisticas(all_results)
        
        # Exibir ranking final
        exibir_ranking(estatisticas)
        
        print(f"\n✅ EXPERIMENTO CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
    except Exception as e:
        print(f"❌ ERRO durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
