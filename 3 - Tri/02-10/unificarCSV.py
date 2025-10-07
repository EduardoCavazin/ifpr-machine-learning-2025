import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

BASEDIR = 'trabalhos'
RESULT = 'all.csv'

MYCOLS = ['dataset', 'classifier', 'metric', 
         'v1', 'v2', 'v3', 'v4', 
         'v5', 'v6', 'v7', 'v8', 
         'v9', 'v10', 'v11', 'v12', 
         'v13', 'v14', 'v15', 'v16', 
         'v17', 'v18', 'v19', 'v20', 
         'author']

# Dicionários de padronização
CLASSIFIER_MAPPING = {
    'perceptron': 'perceptron',
    'Perceptron': 'perceptron',
    ' perceptron': 'perceptron',
    'svm': 'svm',
    'SVM': 'svm',
    'SVC': 'svm',
    ' svm': 'svm',
    'bayes': 'naive_bayes',
    'NaiveBayes': 'naive_bayes',
    'Naive Bayes': 'naive_bayes',
    'GaussianNB': 'naive_bayes',
    ' bayes': 'naive_bayes',
    'trees': 'decision_tree',
    'DecisionTree': 'decision_tree',
    'Decision Tree': 'decision_tree',
    ' trees': 'decision_tree',
    'knn': 'knn',
    'KNN': 'knn',
    'KNeighbors': 'knn',
    ' knn': 'knn',
    'LogisticRegression': 'logistic_regression',
    'RandomForest': 'random_forest'
}

METRIC_MAPPING = {
    'f1': 'f1_score',
    'F1': 'f1_score',
    'f1-score': 'f1_score',
    'F1-Score': 'f1_score',
    'f1_score': 'f1_score',
    'F1_Score': 'f1_score',
    'F1-Measure': 'f1_score',
    ' f1': 'f1_score',
    'accuracy': 'accuracy',
    'Accuracy': 'accuracy',
    'Acurácia': 'accuracy',
    'Acuracia': 'accuracy',
    'acc': 'accuracy',
    'ACC': 'accuracy',
    'Acc': 'accuracy',
    ' acc': 'accuracy'
}


def standardize_data(df):
    """Padroniza os valores das colunas classifier e metric"""
    if 'classifier' in df.columns:
        df['classifier'] = df['classifier'].map(CLASSIFIER_MAPPING).fillna(df['classifier'])
    if 'metric' in df.columns:
        df['metric'] = df['metric'].map(METRIC_MAPPING).fillna(df['metric'])
    return df

mylist = os.listdir(BASEDIR)
result = []
errors = []

for fname in mylist:
    if not fname.endswith('.csv'):
        continue
    
    try:
        print(f"Processando: {fname}")
        filepath = os.path.join(BASEDIR, fname)
        df = pd.read_csv(filepath)
        
        # Validação do número de colunas
        if len(df.columns) != len(MYCOLS) - 1:  # -1 porque 'author' será adicionada
            print(f"AVISO: {fname} tem {len(df.columns)} colunas, esperado {len(MYCOLS)-1}")
        
        df['author'] = fname[:-4]
        
        # Ajusta colunas se necessário
        if len(df.columns) == len(MYCOLS):
            df.columns = MYCOLS
        
        # Padroniza dados
        df = standardize_data(df)
        result.append(df)
        
    except Exception as e:
        errors.append(f"Erro ao processar {fname}: {str(e)}")
        print(f"ERRO: {fname} - {str(e)}")

if result:
    final_result = pd.concat(result, axis=0, ignore_index=True)
    final_result.to_csv(RESULT, index=False)
    print(f"\nArquivo {RESULT} criado com sucesso!")
    print(f"Total de registros: {len(final_result)}")
    print(f"Arquivos processados: {len(result)}")
    
    if errors:
        print(f"\nErros encontrados: {len(errors)}")
        for error in errors:
            print(f"- {error}")
else:
    print("Nenhum arquivo foi processado com sucesso!")
    
    
# Primeiro, vamos carregar os dados processados
df_final = pd.read_csv(RESULT)

print("=== ANÁLISE DOS DADOS ===")
print(f"Total de registros: {len(df_final)}")
print(f"Classificadores únicos: {df_final['classifier'].unique()}")
print(f"Métricas únicas: {df_final['metric'].unique()}")
print()


# Filtrar apenas f1 + naive_bayes
filtered_data = df_final[
    (df_final['classifier'] == 'naive_bayes') & 
    (df_final['metric'] == 'f1_score')
].copy()

print("=== DADOS FILTRADOS (Naive Bayes + F1 Score) ===")
print(f"Registros encontrados: {len(filtered_data)}")
print(f"Datasets únicos: {filtered_data['dataset'].unique()}")
print(f"Autores únicos: {filtered_data['author'].unique()}")
print()


# Identificar datasets únicos nos dados filtrados
datasets_unicos = filtered_data['dataset'].unique()
print(f"=== DATASETS ENCONTRADOS ===")
for i, dataset in enumerate(datasets_unicos, 1):
    print(f"Dataset {i}: {dataset}")
print()

# Preparar dados para o boxplot agrupando por dataset (cada linha é um dataset)
plot_data = []
plot_labels = []
value_columns = [f'v{i}' for i in range(1, 21)]

for dataset in datasets_unicos:
    # Filtrar dados para este dataset específico
    dataset_rows = filtered_data[filtered_data['dataset'] == dataset]
    
    # Coletar todos os valores das 20 colunas para este dataset
    dataset_values = []
    for _, row in dataset_rows.iterrows():
        for col in value_columns:
            if pd.notna(row[col]):  # Só adicionar valores não-nulos
                try:
                    val = float(row[col])
                    dataset_values.append(val)
                except (ValueError, TypeError):
                    continue
    
    if len(dataset_values) > 0:
        plot_data.append(dataset_values)
        plot_labels.append(dataset)
        print(f"Dataset '{dataset}': {len(dataset_values)} valores coletados")

print()

# Criar o boxplot com nomes dos datasets
plt.figure(figsize=(18, 10))
plt.boxplot(plot_data, labels=plot_labels)
plt.title('Boxplot: Naive Bayes - F1 Score\nDistribuição dos Valores por Dataset', 
          fontsize=16, fontweight='bold')
plt.xlabel('Datasets', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Estatísticas descritivas por dataset (baseado nas linhas, não colunas)
print("=== ESTATÍSTICAS DESCRITIVAS POR DATASET ===")
value_columns = [f'v{i}' for i in range(1, 21)]

for dataset in datasets_unicos:
    dataset_rows = filtered_data[filtered_data['dataset'] == dataset]
    
    # Coletar todos os valores das 20 colunas para este dataset
    dataset_values = []
    for _, row in dataset_rows.iterrows():
        for col in value_columns:
            if pd.notna(row[col]):
                try:
                    val = float(row[col])
                    dataset_values.append(val)
                except (ValueError, TypeError):
                    continue
    
    if len(dataset_values) > 0:
        dataset_values = np.array(dataset_values)
        print(f"{dataset}: Média={dataset_values.mean():.4f}, "
              f"Mediana={np.median(dataset_values):.4f}, "
              f"Std={dataset_values.std():.4f}, "
              f"Min={dataset_values.min():.4f}, "
              f"Max={dataset_values.max():.4f}")

# Análise de outliers por dataset
print("\n=== ANÁLISE DE OUTLIERS POR DATASET ===")
for dataset in datasets_unicos:
    dataset_rows = filtered_data[filtered_data['dataset'] == dataset]
    
    # Coletar todos os valores das 20 colunas para este dataset
    dataset_values = []
    for _, row in dataset_rows.iterrows():
        for col in value_columns:
            if pd.notna(row[col]):
                try:
                    val = float(row[col])
                    dataset_values.append(val)
                except (ValueError, TypeError):
                    continue
    
    if len(dataset_values) > 0:
        dataset_values = np.array(dataset_values)
        q1 = np.percentile(dataset_values, 25)
        q3 = np.percentile(dataset_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = dataset_values[(dataset_values < lower_bound) | (dataset_values > upper_bound)]
        
        if len(outliers) > 0:
            print(f"{dataset}: {len(outliers)} outliers encontrados -> {outliers}")
        else:
            print(f"{dataset}: Nenhum outlier (Range: {lower_bound:.4f} a {upper_bound:.4f})")

print("\n=== RESUMO GERAL DOS DADOS ===")
all_values = []
for col in value_columns:
    valid_values = filtered_data[col].dropna()
    if valid_values.dtype == 'O':
        valid_values = pd.to_numeric(valid_values, errors='coerce')
    all_values.extend(valid_values.values)

if all_values:
    all_values = np.array(all_values)
    print(f"Total de valores: {len(all_values)}")
    print(f"Média geral: {all_values.mean():.4f}")
    print(f"Desvio geral: {all_values.std():.4f}")
    print(f"Range geral: {all_values.min():.4f} a {all_values.max():.4f}")
    
    # Verificar distribuição
    q1_geral = np.percentile(all_values, 25)
    q3_geral = np.percentile(all_values, 75)
    iqr_geral = q3_geral - q1_geral
    print(f"IQR geral: {iqr_geral:.4f} (Q1={q1_geral:.4f}, Q3={q3_geral:.4f})")
    
    outliers_gerais = all_values[(all_values < q1_geral - 1.5*iqr_geral) | (all_values > q3_geral + 1.5*iqr_geral)]
    print(f"Outliers gerais: {len(outliers_gerais)} de {len(all_values)} ({len(outliers_gerais)/len(all_values)*100:.1f}%)")
    
    if len(outliers_gerais) > 0:
        print(f"Valores outliers: {sorted(outliers_gerais)}")
print()

# Boxplot alternativo usando seaborn (mais bonito)
plt.figure(figsize=(15, 8))

# Converter dados para formato long para seaborn - agrupando por dataset nas linhas
long_data = []
value_columns = [f'v{i}' for i in range(1, 21)]

for dataset in datasets_unicos:
    dataset_rows = filtered_data[filtered_data['dataset'] == dataset]
    
    # Para cada linha deste dataset, coletar os 20 valores
    for _, row in dataset_rows.iterrows():
        for col in value_columns:
            if pd.notna(row[col]):
                try:
                    val = float(row[col])
                    long_data.append({'Dataset': dataset, 'F1 Score': val})
                except (ValueError, TypeError):
                    continue

long_df = pd.DataFrame(long_data)

if len(long_df) > 0:
    plt.figure(figsize=(18, 10))
    sns.boxplot(data=long_df, x='Dataset', y='F1 Score')
    plt.title('Boxplot: Naive Bayes - F1 Score\n(Usando Seaborn)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Datasets', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Mostrar resumo final
    print(f"\n=== RESUMO FINAL ===")
    print(f"Classificador analisado: Naive Bayes")
    print(f"Métrica analisada: F1 Score")
    print(f"Total de datasets analisados: {len(datasets_unicos)}")
    print(f"Datasets: {datasets_unicos}")
    print(f"Total de valores analisados: {len(long_df)}")
    
    # Verificar completude dos dados
    total_values_expected = len(datasets_unicos) * 20  # 20 valores por dataset
    if len(long_df) < total_values_expected:
        missing_count = total_values_expected - len(long_df)
        print(f"⚠️  Atenção: {missing_count} valores faltantes de {total_values_expected} esperados")
    else:
        print("✅ Todos os dados estão completos!")
else:
    print("❌ Nenhum dado válido encontrado para Naive Bayes + F1 Score")
    
    
# Gráfico final - Salvar com nomes dos datasets
plt.figure(figsize=(20, 10))
sns.boxplot(data=long_df, x='Dataset', y='F1 Score', color='lightblue')
plt.title('Naive Bayes - F1 Score\nDistribuição por Dataset', 
          fontsize=18, fontweight='bold')
plt.xlabel('Datasets', fontsize=16)
plt.ylabel('F1 Score', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Salvar o gráfico com nomes dos datasets
plt.savefig('naive_bayes_f1_score_por_dataset.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Gráfico salvo como 'naive_bayes_f1_score_por_dataset.png'")