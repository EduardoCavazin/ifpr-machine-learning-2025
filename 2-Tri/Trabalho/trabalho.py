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
NUM_EXECUCOES_PRINCIPAIS = 3    # Quantas execuções principais
NUM_FOLDS_CV = 5               # Quantos folds no cross validation
NUM_EXECUCOES_POR_FOLD = 1    # Quantas execuções por fold 
TRAIN_RATIO = 0.8              # Proporção treino/teste (80%/20%)
LIMIAR_ZEROS = 0.5             # Limiar para remoção de colunas com maioria zeros (50%)

# Random state global
rng = np.random.RandomState()


def carregar_dataset_adult():
    print("Carregando dataset Adult (Census Income)...")
    
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    df = pd.read_csv('adult.csv', names=column_names, skipinitialspace=True)
    
    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
    print(f"Classes target: {df['income'].value_counts().to_dict()}")
    
    return df


def processar_dataset(df):
    print("\nProcessando dataset...")
    
    linhas_antes = len(df)
    
    df_clean = df.replace('?', pd.NA).dropna()
    linhas_depois = len(df_clean)
    
    print(f"   Dados faltantes removidos: {linhas_antes - linhas_depois} linhas")
    print(f"   Dataset final: {linhas_depois} amostras")
    
    df_processed = df_clean.copy()
    
    print("   Processando coluna target 'income'...")
    df_processed['income'] = df_processed['income'].map({'<=50K': 0, '>50K': 1})
    
    categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
    if 'income' in categorical_columns:
        categorical_columns.remove('income')
    
    print(f"   Convertendo {len(categorical_columns)} colunas categóricas:")
    for col in categorical_columns:
        print(f"     - {col}: {df_processed[col].nunique()} categorias")
    
    le_dict = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        le_dict[col] = le

    print(f"Removendo colunas com maioria de zeros (limiar: {LIMIAR_ZEROS*100:.0f}%)...")
    
    features_cols = [col for col in df_processed.columns if col != 'income']
    colunas_para_remover = []
    
    for col in features_cols:
        total_valores = len(df_processed[col])
        zeros_count = (df_processed[col] == 0).sum()
        proporcao_zeros = zeros_count / total_valores
        
        if proporcao_zeros > LIMIAR_ZEROS:
            colunas_para_remover.append(col)
            print(f"      {col}: {proporcao_zeros*100:.1f}% zeros ({zeros_count}/{total_valores}) - REMOVIDA")
        else:
            print(f"      {col}: {proporcao_zeros*100:.1f}% zeros ({zeros_count}/{total_valores}) - MANTIDA")
    
    if colunas_para_remover:
        df_processed = df_processed.drop(columns=colunas_para_remover)
        print(f"    {len(colunas_para_remover)} colunas removidas por excesso de zeros")
    else:
        print(f"    Nenhuma coluna removida - todas dentro do limiar")
    
    print(f"    Dataset final: {df_processed.shape[1]-1} features + 1 target")
    
    # ===== NORMALIZAÇÃO MIN-MAX =====
    print("   Aplicando normalização Min-Max...")
    target_column = 'income'
    X = df_processed.drop(columns=[target_column]).values
    y = df_processed[target_column].values
    
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    print(f"   Features normalizadas para intervalo [0,1]")
    print(f"   Distribuição target: Classe 0 (≤50K): {np.sum(y == 0)}, Classe 1 (>50K): {np.sum(y == 1)}")
    print("Dataset processado!")
    
    return X_normalized, y


def preparar_dados_ml(df):
    print(f"\n Preparando dados para Cross Validation...")
    print(f"   Configuração: {NUM_EXECUCOES_PRINCIPAIS} execuções × {NUM_FOLDS_CV} folds × {NUM_EXECUCOES_POR_FOLD} exec/fold")
    print(f"   Divisão treino/teste: {TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}%")
    print(f"   Limiar zeros: {LIMIAR_ZEROS*100:.0f}% (colunas com mais zeros que isso são removidas)")
    print(f"   Total de avaliações por algoritmo: {NUM_EXECUCOES_PRINCIPAIS * NUM_FOLDS_CV * NUM_EXECUCOES_POR_FOLD}")
    return df


# ===== CROSS VALIDATION MANUAL =====
def get_cv_value(X_data, y_data):
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
        
        xtr = X_data[:part]  
        ytr = y_data[:part]    
        xte = X_data[part:] 
        yte = y_data[part:] 
        
        fold_results = {
            'perceptron': [],
            'svm': [],
            'bayes': [],
            'trees': [],
            'knn': []
        }
        
        for exec_fold in range(NUM_EXECUCOES_POR_FOLD):
            if NUM_EXECUCOES_POR_FOLD > 1:
                print(f"     Exec {exec_fold + 1}/{NUM_EXECUCOES_POR_FOLD}:")
            
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
                
                fim = time.time()
                tempo = fim - inicio
                
                fold_results[clf_name].append(f1)
                exec_scores.append(f1)
                exec_tempos.append(tempo)
            
            if NUM_EXECUCOES_POR_FOLD == 1:
                
                for i, clf_name in enumerate(fold_results.keys()):
                    print(f"     {clf_name:<12}: F1 = {exec_scores[i]:.4f} | Tempo = {exec_tempos[i]:.3f}s")
            else:
                
                if exec_fold == 0:  #  Mostrar cabeçalho apenas na primeira execução
                    algoritmos_nomes = list(fold_results.keys())
                    cabecalho_scores = " | ".join([f"{nome:>8}" for nome in algoritmos_nomes])
                    cabecalho_tempos = " | ".join([f"{nome[:8]:>6}s" for nome in algoritmos_nomes])
                    print(f"     {'Ex':<2}: {cabecalho_scores} | {cabecalho_tempos}")
                    print(f"     {'-'*2}: {'-'*8*len(algoritmos_nomes) + '-'*(len(algoritmos_nomes)-1)} | {'-'*6*len(algoritmos_nomes) + '-'*(len(algoritmos_nomes)-1)}")
                
                scores_str = " | ".join([f"{score:>8.4f}" for score in exec_scores])
                tempos_str = " | ".join([f"{tempo:>6.3f}s" for tempo in exec_tempos])
                print(f"     {exec_fold+1:2d}: {scores_str} | {tempos_str}")
        
        for clf_name in fold_results.keys():
            media_fold = np.mean(fold_results[clf_name])
            results[clf_name].append(media_fold)
        
        if NUM_EXECUCOES_POR_FOLD > 1:
            medias_str = " | ".join([f"{np.mean(fold_results[nome]):>8.4f}" for nome in fold_results.keys()])
            print(f"     {'Média':<2}: {medias_str}")
            print()  
        else:
            print() 
        
        y_data = list(y_data[part:]) + list(y_data[:part])
        X_data = list(X_data[part:]) + list(X_data[:part])
    
    print(f"    Médias dos {NUM_FOLDS_CV} folds:")
    fold_results = {}
    for clf_name, result in results.items():
        media = sum(result) / len(result)
        fold_results[clf_name] = media
        print(f"     {clf_name:<12}: {media:.4f}")
    
    return fold_results


def executar_experimentos(X, y):
    print(f"\n EXECUTANDO {NUM_EXECUCOES_PRINCIPAIS} EXECUÇÕES DE CROSS VALIDATION")
    print("="*80)
    
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
        
        idx = list(range(len(y)))
        shuffle(idx)
        X_shuffled = X[idx]
        y_shuffled = y[idx]
        
        fold_results = get_cv_value(X_shuffled, y_shuffled)
        
        for clf_name, result in fold_results.items():
            all_results[clf_name].append(result)
        
        print(f"    Execução {exec_id + 1} concluída!")
    
    warnings.filterwarnings('default')
    
    return all_results


def calcular_estatisticas(all_results):
    print(f"\n ESTATÍSTICAS FINAIS - {NUM_EXECUCOES_PRINCIPAIS} EXECUÇÕES")
    print("="*70)
    
    estatisticas = {}
    
    print(f"{'Algoritmo':<12} | {'Exec 1':<8} | {'Exec 2':<8} | {'Exec 3':<8} | {'Média Final':<10} | {'Desvio':<8}")
    print("-" * 68)
    
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
            print(f"{clf_name:<12} | {scores[0]:<8.4f} | {scores[1]:<8.4f} | {scores[2]:<8.4f} | {media_final:<10.4f} | {desvio:<8.4f}")
        else:
            scores_str = " | ".join([f"{score:<8.4f}" for score in scores])
            print(f"{clf_name:<12} | {scores_str:<26} | {media_final:<10.4f} | {desvio:<8.4f}")
    
    return estatisticas


def exibir_ranking(estatisticas):
    print(f"\n RANKING FINAL - DATASET ADULT")
    print("="*50)
    
    ranking = sorted(estatisticas.items(), key=lambda x: x[1]['media_final'], reverse=True)
    
    medalhas = [" 1º", " 2º", " 3º", "  4º", "  5º"]
    
    for i, (nome, stats) in enumerate(ranking):
        medalha = medalhas[i] if i < len(medalhas) else f"   {i+1}º"
        print(f"{medalha} {nome:<12}: {stats['media_final']:.4f} (±{stats['desvio']:.4f})")
    
    total_avaliacoes = NUM_EXECUCOES_PRINCIPAIS * NUM_FOLDS_CV * NUM_EXECUCOES_POR_FOLD
    print(f"\nResumo: {NUM_EXECUCOES_PRINCIPAIS} exec × {NUM_FOLDS_CV} folds × {NUM_EXECUCOES_POR_FOLD} exec/fold = {total_avaliacoes} avaliações por algoritmo")
    print("Target: Predição de renda (≤50K vs >50K)")


def main():
    print("TRABALHO: COMPARAÇÃO DE ALGORITMOS ML - DATASET ADULT")
    print("="*70)
    print(f"  Configuração: {NUM_EXECUCOES_PRINCIPAIS} exec × {NUM_FOLDS_CV} folds × {NUM_EXECUCOES_POR_FOLD} exec/fold × {TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}% split")
    print(f" Total de avaliações por algoritmo: {NUM_EXECUCOES_PRINCIPAIS * NUM_FOLDS_CV * NUM_EXECUCOES_POR_FOLD}")
    print(f" Remoção de colunas com >{LIMIAR_ZEROS*100:.0f}% zeros ativada")
    print("="*70)
    
    try:
        df = carregar_dataset_adult()
        
        X, y = processar_dataset(df)
        
        preparar_dados_ml(X)
        
        all_results = executar_experimentos(X, y)
        
        estatisticas = calcular_estatisticas(all_results)
        
        exibir_ranking(estatisticas)
        
        print(f"\n TRABALHO CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
    except Exception as e:
        print(f" ERRO durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()