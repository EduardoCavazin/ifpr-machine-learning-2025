import pandas as pd
import numpy as np

#Fazer uam função que retorne um dicionário que retorne uma chave "dataset", possuindo todo o dataset, menos
#a coluna da classe, e uma chave "classe", possuindo as classes como chaves e os valores transformados em
#números inteiros

def data_set(dataset_path):
    
    result = {}
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
        ]
    
    df = pd.read_csv(dataset_path, names=column_names, skipinitialspace=True)
    
    df = df.replace('?', np.nan)
    df = df.dropna()
    
    class_columns = df.columns[-1]
    
    features_df = df.drop(columns=[class_columns]).copy()
    
    for col in features_df.select_dtypes(include=['object']).columns:
        unique_values = features_df[col].unique()
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        features_df[col] = features_df[col].map(mapping)

    class_values = df[class_columns]
    unique_classes = class_values.unique()
    class_mapping = {class_name: idx for idx, class_name in enumerate(unique_classes)}

    encoded_classes = class_values.map(class_mapping).values
    
    result['dataset'] = features_df
    result['classe'] = encoded_classes
    
    result['class_mapping'] = class_mapping
    result['class_names'] = unique_classes
    
    return result

def save_processed_data(data, output_path='adult_processed.csv'):
    complete_df = data['dataset'].copy()
    complete_df['income_encoded'] = data['classe']
    
    complete_df.to_csv(output_path, index=False)
    print(f"Dataset processado salvo em: {output_path}")
    print(f"Shape do arquivo salvo: {complete_df.shape}")
    
    mappings_path = output_path.replace('.csv', '_mappings.txt')
    with open(mappings_path, 'w') as f:
        f.write("MAPEAMENTOS DO DATASET ADULT PROCESSADO\n")
        f.write("="*50 + "\n\n")
        
        f.write("MAPEAMENTO DAS CLASSES:\n")
        f.write("-"*30 + "\n")
        for class_name, class_code in data['class_mapping'].items():
            f.write(f"{class_name} -> {class_code}\n")
        
        f.write(f"\nCOLUNAS DO DATASET FINAL:\n")
        f.write("-"*30 + "\n")
        for i, col in enumerate(complete_df.columns):
            f.write(f"{i+1}. {col}\n")
        
        f.write(f"\nINFORMAÇÕES GERAIS:\n")
        f.write("-"*30 + "\n")
        f.write(f"Total de linhas: {len(complete_df)}\n")
        f.write(f"Total de colunas: {len(complete_df.columns)}\n")
        f.write(f"Arquivo original processado: adult.csv\n")
    
    print(f"Mapeamentos salvos em: {mappings_path}")
    
    return output_path
    


DATASET_PATH = 'adult.csv'

if __name__ == "__main__":
    data = data_set(DATASET_PATH)
    
    print("="*50)
    print("PROCESSAMENTO DO DATASET ADULT")
    print("="*50)
    
    print(f"Dataset shape: {data['dataset'].shape}")
    print(f"Classes shape: {data['classe'].shape}")
    
    print("="*50)
    
    print(f"\nPrimeiras 5 linhas do dataset:")
    print(data['dataset'].head())
    
    print("="*50)

    print(f"\nMapeamento das classes:")
    for class_name, class_code in data['class_mapping'].items():
        print(f"  {class_name} -> {class_code}")
        
    print("="*50)
        
    print(f"\nDistribuição das classes:")
    unique, counts = np.unique(data['classe'], return_counts=True)
    for class_code, count in zip(unique, counts):
        class_name = data['class_names'][class_code]
        print(f"  {class_name} ({class_code}): {count} amostras")
        
    print("="*50)
        
    print(f"\nTipos de dados do dataset:")
    print(data['dataset'].dtypes.value_counts())
    
    print("="*50)
    
    save_processed_data(data, output_path='adult_processed.csv')
    
    print("\n" + "="*50)
    print("PROCESSAMENTO COMPLETO!")
    print("="*50)
    print("Arquivos gerados:")
    print("1. adult_processed.csv - Dataset processado")
    print("2. adult_processed_mappings.txt - Informações dos mapeamentos")