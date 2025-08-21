"""
===============================================================================
EXERCÍCIO: PROCESSAMENTO DATASET AUTOMOBILE + MACHINE LEARNING (PERCEPTRON/SVC)
===============================================================================
Procedimentos:
1. Carregar dataset
2. Remover dados faltantes
3. Processar dados categóricos
4. Fazer shuffle dos dados
5. Dividir 60% treino / 40% teste
6. Escolher algoritmo (Perceptron ou SVC)
7. Treinar algoritmo escolhido múltiplas vezes
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from random import shuffle
import time
import warnings


def exibir_matriz_confusao(y_true, y_pred, classes_unicas):
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print("\n📋 MATRIZ DE CONFUSÃO:")
    print("="*50)
    
    # Cabeçalho das colunas
    print("Real\\Predito", end="")
    for classe in classes_unicas:
        print(f"{classe:>8}", end="")
    print()
    
    # Linha de separação
    print("-" * (10 + 8 * len(classes_unicas)))
    
    # Dados da matriz
    for i, classe_real in enumerate(classes_unicas):
        print(f"{classe_real:>9}", end=" ")
        for j, classe_pred in enumerate(classes_unicas):
            print(f"{conf_matrix[i][j]:>7}", end=" ")
        print()
    
    print("="*50)
    
    return conf_matrix


def carregar_dataset_automobile():
    print(" Carregando dataset automobile...")
    
    # Definir nomes das colunas baseado na documentação
    column_names = [
        'symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration',
        'num_of_doors', 'body_style', 'drive_wheels', 'engine_location',
        'wheel_base', 'length', 'width', 'height', 'curb_weight',
        'engine_type', 'num_of_cylinders', 'engine_size', 'fuel_system',
        'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm',
        'city_mpg', 'highway_mpg', 'price'
    ]
    
    # Carregar dataset
    df = pd.read_csv('../../Assets/automobile/imports-85.csv', 
                     names=column_names, na_values='?')
    
    print(f" Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
    
    return df


def analisar_dados_faltantes(df):
    print("\n ANÁLISE DE DADOS FALTANTES")
    print("="*50)
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Coluna': missing_data.index,
        'Faltantes': missing_data.values,
        'Percentual': missing_percent.values
    })
    
    # Mostrar apenas colunas com dados faltantes
    missing_df = missing_df[missing_df['Faltantes'] > 0].sort_values('Faltantes', ascending=False)
    
    if len(missing_df) > 0:
        print("Colunas com dados faltantes:")
        print(missing_df.to_string(index=False))
        print(f"\nTotal de linhas com dados faltantes: {df.isnull().any(axis=1).sum()}")
    else:
        print(" Nenhum dado faltante encontrado!")
    
    return missing_df


def remover_dados_faltantes(df):
    print(f"\n Removendo dados faltantes...")
    linhas_antes = len(df)
    
    # Remover linhas com qualquer valor NaN
    df_clean = df.dropna()
    
    linhas_depois = len(df_clean)
    linhas_removidas = linhas_antes - linhas_depois
    
    print(f"   Linhas antes: {linhas_antes}")
    print(f"   Linhas depois: {linhas_depois}")
    print(f"   Linhas removidas: {linhas_removidas} ({(linhas_removidas/linhas_antes)*100:.1f}%)")
    
    return df_clean


def processar_dados_categoricos(df):
    print(f"\n Processando dados categóricos...")
    
    df_processed = df.copy()
    categorical_mappings = {}
    
    # Identificar colunas categóricas
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    print(f"Colunas categóricas encontradas: {len(categorical_columns)}")
    
    le = LabelEncoder()
    for col in categorical_columns:
        print(f"   Convertendo: {col}")
        unique_values = df_processed[col].unique()
        print(f"      Valores únicos: {len(unique_values)} - {list(unique_values)[:5]}...")
        
        # Aplicar Label Encoder
        df_processed[col] = le.fit_transform(df_processed[col])
        
        # Salvar mapeamento para referência
        categorical_mappings[col] = dict(zip(unique_values, le.transform(unique_values)))
    
    print(f" Conversão concluída!")
    
    return df_processed, categorical_mappings


def preparar_dados_para_ml(df):
    print(f"\nPreparando dados para ML...")
    
    # Usar 'symboling' como variável target (risco do veículo)
    target_column = 'symboling'
    
    # Separar features e target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Classes únicas no target: {sorted(y.unique())}")
    print(f"   Distribuição das classes:")
    
    class_counts = y.value_counts().sort_index()
    for class_val, count in class_counts.items():
        print(f"      Classe {class_val}: {count} amostras ({count/len(y)*100:.1f}%)")
    
    return X.values, y.values


def embaralhar_e_dividir_dados(X, y, train_ratio=0.6):
    print(f"\nEmbaralhando e dividindo dados...")
    print(f"   Treino: {train_ratio*100:.0f}% | Teste: {(1-train_ratio)*100:.0f}%")
    
    # Criar índices e embaralhar
    indices = list(range(len(y)))
    shuffle(indices)
    
    # Aplicar embaralhamento
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Dividir dados
    split_point = int(len(y) * train_ratio)
    
    X_train = X_shuffled[:split_point]
    y_train = y_shuffled[:split_point]
    X_test = X_shuffled[split_point:]
    y_test = y_shuffled[split_point:]
    
    print(f"      Divisão concluída:")
    print(f"      Treino: {len(X_train)} amostras")
    print(f"      Teste: {len(X_test)} amostras")
    
    return X_train, X_test, y_train, y_test


def treinar_algoritmo_multiplas_vezes(X_train, X_test, y_train, y_test, algoritmo='perceptron', num_execucoes=20, normalizar=True):
    if algoritmo.lower() == 'perceptron':
        algoritmo_nome = "PERCEPTRON"
        modelo_config = lambda i: Perceptron(max_iter=1000, random_state=i)
    elif algoritmo.lower() == 'svc':
        algoritmo_nome = "SVC"
        modelo_config = lambda i: SVC(random_state=i)
    else:
        raise ValueError("Algoritmo deve ser 'perceptron' ou 'svc'")
    
    print(f"\nTREINANDO {algoritmo_nome} - {num_execucoes} EXECUÇÕES")
    print("="*60)
    
    
    X_train_processed = X_train
    X_test_processed = X_test
    print("Usando dados sem normalização")
    
    acuracias = []
    tempos = []
    precisions_macro = []
    recalls_macro = []
    f1scores_macro = []
    
    print(f"\nIniciando {num_execucoes} execuções...")
    print("Exec | Acurácia | Tempo(s) | Precisão | Recall | F1-Score")
    print("-----|----------|----------|----------|--------|----------")
    
    # Suprimir warnings temporariamente
    warnings.filterwarnings('ignore')
    
    for i in range(num_execucoes):
        start_time = time.time()
        
        modelo = modelo_config(i)
        
        modelo.fit(X_train_processed, y_train)
        
        y_pred = modelo.predict(X_test_processed)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calcular métricas com zero_division para evitar warnings
        precision, recall, f1score, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        
        acuracias.append(accuracy)
        tempos.append(training_time)
        precisions_macro.append(precision)
        recalls_macro.append(recall)
        f1scores_macro.append(f1score)
        
        print(f" {i+1:2d}  | {accuracy:.4f}   | {training_time:.3f}    | {precision:.4f}   | {recall:.4f}  | {f1score:.4f}")
    
    # Restaurar warnings
    warnings.filterwarnings('default')
    
    media_acc = np.mean(acuracias)
    desvio_acc = np.std(acuracias)
    min_acc = np.min(acuracias)
    max_acc = np.max(acuracias)
    
    media_precision = np.mean(precisions_macro)
    media_recall = np.mean(recalls_macro)
    media_f1score = np.mean(f1scores_macro)
    
    tempo_medio = np.mean(tempos)
    tempo_total = np.sum(tempos)
    
    print(f"\nESTATÍSTICAS FINAIS")
    print("="*50)
    print(f"ACURÁCIA:")
    print(f"   Média:         {media_acc:.4f} ({media_acc*100:.2f}%)")
    print(f"   Desvio padrão: {desvio_acc:.4f}")
    print(f"   Mínimo:        {min_acc:.4f} ({min_acc*100:.2f}%)")
    print(f"   Máximo:        {max_acc:.4f} ({max_acc*100:.2f}%)")
    
    print(f"\nMÉTRICAS MACRO (Média das {num_execucoes} execuções):")
    print(f"   Precisão:      {media_precision:.4f}")
    print(f"   Recall:        {media_recall:.4f}")
    print(f"   F1-Score:      {media_f1score:.4f}")
    
    print(f"\nTEMPO:")
    print(f"   Tempo médio:   {tempo_medio:.3f}s")
    print(f"   Tempo total:   {tempo_total:.3f}s")
    
    # Treinar modelo final para matriz de confusão (usando melhor random_state)
    melhor_indice = np.argmax(acuracias)
    print(f"\nMelhor execução: #{melhor_indice+1} (Acurácia: {acuracias[melhor_indice]:.4f})")
    
    # Treinar modelo final
    modelo_final = modelo_config(melhor_indice)
    modelo_final.fit(X_train_processed, y_train)
    y_pred_final = modelo_final.predict(X_test_processed)
    
    # Mostrar matriz de confusão do melhor modelo
    classes_unicas = sorted(np.unique(y_test))
    conf_matrix = exibir_matriz_confusao(y_test, y_pred_final, classes_unicas)
    
    # Relatório simplificado (só métricas principais)
    print(f"\nRELATÓRIO SIMPLIFICADO (Melhor Modelo):")
    print("="*50)
    
    # Suprimir warnings temporariamente
    warnings.filterwarnings('ignore')
    
    # Calcular métricas do melhor modelo
    precision_final, recall_final, f1score_final, _ = precision_recall_fscore_support(
        y_test, y_pred_final, average='macro', zero_division=0
    )
    
    print(f"Acurácia:  {acuracias[melhor_indice]:.4f} ({acuracias[melhor_indice]*100:.2f}%)")
    print(f"Precisão:  {precision_final:.4f}")
    print(f"Recall:    {recall_final:.4f}")
    print(f"F1-Score:  {f1score_final:.4f}")
    
    # Restaurar warnings
    warnings.filterwarnings('default')
 
    return  media_acc, acuracias


def main():
    print("EXERCÍCIO: DATASET AUTOMOBILE + MACHINE LEARNING")
    print("="*70)
    
    algoritmo = input("Escolha o algoritmo (perceptron/svc): ").strip().lower()
    if algoritmo not in ['perceptron', 'svc']:
        print("Algoritmo não reconhecido. Usando Perceptron como padrão.")
        algoritmo = 'perceptron'
    
    print(f" Algoritmo selecionado: {algoritmo.upper()}")
    print("="*70)
    
    try:
        # 1. Carregar dataset
        df = carregar_dataset_automobile()
        
        # 2. Analisar dados faltantes
        analisar_dados_faltantes(df)
        
        # 3. Remover dados faltantes
        df_clean = remover_dados_faltantes(df)
        
        # 4. Processar dados categóricos
        df_processed, mappings = processar_dados_categoricos(df_clean)
        
        # 5. Preparar dados para ML
        X, y = preparar_dados_para_ml(df_processed)
        
        # 6. Embaralhar e dividir dados (60% treino, 40% teste)
        X_train, X_test, y_train, y_test = embaralhar_e_dividir_dados(X, y, train_ratio=0.6)
        
        # 7. Treinar algoritmo escolhido 20 vezes
        accuracy, todas_acuracias = treinar_algoritmo_multiplas_vezes(
            X_train, X_test, y_train, y_test, algoritmo=algoritmo, num_execucoes=20, normalizar=True
        )
        
        print(f"\nEXERCÍCIO CONCLUÍDO COM SUCESSO!")
        print(f"   Algoritmo usado: {algoritmo.upper()}")
        print(f"   Acurácia média: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Melhor acurácia: {max(todas_acuracias):.4f} ({max(todas_acuracias)*100:.2f}%)")
        print(f"   Pior acurácia: {min(todas_acuracias):.4f} ({min(todas_acuracias)*100:.2f}%)")
        print("="*70)
        
    except Exception as e:
        print(f"ERRO durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
