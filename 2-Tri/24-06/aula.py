import numpy as np
import pandas as pd

def load_adult_dataset():
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    df = pd.read_csv('adult.csv', names=column_names, skipinitialspace=True)
    return df

def remove_missing_data(df):
    print("="*50)
    print("1 -- EXCLUINDO LINHAS COM DADOS FALTANTES")
    print("="*50)
    
    print(f"Dataset original: {df.shape}")
    
    missing_counts = (df == ' ?').sum()
    print("Valores '?' por coluna:")
    print(missing_counts[missing_counts > 0])
    
    df_clean = df.replace(' ?', np.nan)
    df_clean = df_clean.dropna()
    
    print(f"Dataset após limpeza: {df_clean.shape}")
    print(f"Linhas removidas: {len(df) - len(df_clean)}")
    
    return df_clean

def convert_strings_to_numbers(df):
    print("\n" + "="*50)
    print("2 -- TRANSFORMANDO STRINGS EM NÚMEROS")
    print("="*50)
    
    df_encoded = df.copy()
    categorical_mappings = {}
    
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    print(f"Colunas categóricas: {list(categorical_columns)}")
    
    for col in categorical_columns:
        unique_values = df_encoded[col].unique()
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        categorical_mappings[col] = mapping
        df_encoded[col] = df_encoded[col].map(mapping)
        
        print(f"\n{col}:")
        for original, encoded in mapping.items():
            print(f"  {original} -> {encoded}")
    
    print(f"\nTodas as colunas agora são numéricas:")
    print(df_encoded.dtypes)
    
    return df_encoded, categorical_mappings

def separate_age_groups(df):
    print("\n" + "="*50)
    print("3 -- SEPARANDO IDADE POR FAIXAS DE VALORES")
    print("="*50)
    
    df_with_age_groups = df.copy()
    
    age_bins = [0, 25, 35, 45, 55, 65, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    
    df_with_age_groups['age_group'] = pd.cut(
        df_with_age_groups['age'], 
        bins=age_bins, 
        labels=age_labels, 
        include_lowest=True
    )
    
    age_distribution = df_with_age_groups['age_group'].value_counts().sort_index()
    print("Distribuição por faixa etária:")
    print(age_distribution)
    
    print("\nEstatísticas por faixa etária:")
    age_stats = df_with_age_groups.groupby('age_group')['age'].agg(['min', 'max', 'mean', 'count'])
    print(age_stats)
    
    age_group_mapping = {label: idx for idx, label in enumerate(age_labels)}
    df_with_age_groups['age_group_encoded'] = df_with_age_groups['age_group'].map(age_group_mapping)
    
    print(f"\nMapeamento das faixas etárias:")
    for label, code in age_group_mapping.items():
        print(f"  {label} -> {code}")
    
    return df_with_age_groups, age_group_mapping

def show_final_results(df_original, df_final):
    print("\n" + "="*50)
    print("RESULTADOS FINAIS")
    print("="*50)
    
    print(f"Dataset original: {df_original.shape}")
    print(f"Dataset final: {df_final.shape}")
    
    print(f"\nTipos de dados finais:")
    print(df_final.dtypes.value_counts())
    
    print(f"\nPrimeiras 5 linhas do dataset final:")
    print(df_final.head())
    
    print(f"\nColunas do dataset final:")
    print(list(df_final.columns))

def save_processed_data(df, mappings, age_mapping):
    print("\n" + "="*50)
    print("SALVANDO DADOS PROCESSADOS")
    print("="*50)
    
    df.to_csv('adult_processed.csv', index=False)
    print("Dataset processado salvo em: adult_processed.csv")
    
    with open('mappings.txt', 'w') as f:
        f.write("MAPEAMENTOS DAS CONVERSÕES\n")
        f.write("="*50 + "\n\n")
        
        f.write("VARIÁVEIS CATEGÓRICAS:\n")
        f.write("-"*30 + "\n")
        for col, mapping in mappings.items():
            f.write(f"\n{col}:\n")
            for original, encoded in mapping.items():
                f.write(f"  {original} -> {encoded}\n")
        
        f.write(f"\nFAIXAS ETÁRIAS:\n")
        f.write("-"*30 + "\n")
        for label, code in age_mapping.items():
            f.write(f"  {label} -> {code}\n")
    
    print("Mapeamentos salvos em: mappings.txt")

def main():
    print("PROCESSAMENTO DO DATASET ADULT")
    print("="*50)
    
    # 0 -- Carregar dataset adult
    print("0 -- CARREGANDO DATASET ADULT")
    df_original = load_adult_dataset()
    print(f"Dataset carregado: {df_original.shape}")
    
    # 1 -- Excluir linhas com dados faltantes
    df_clean = remove_missing_data(df_original)
    
    # 2 -- Transformar strings em números
    df_encoded, categorical_mappings = convert_strings_to_numbers(df_clean)
    
    # 3 -- Separar idade por faixa de valores
    df_final, age_mapping = separate_age_groups(df_encoded)
    
    # Mostrar resultados finais
    show_final_results(df_original, df_final)
    
    # Salvar dados processados
    save_processed_data(df_final, categorical_mappings, age_mapping)
    
    print("\n" + "="*50)
    print("PROCESSAMENTO CONCLUÍDO!")
    print("="*50)

if __name__ == '__main__':
    main()