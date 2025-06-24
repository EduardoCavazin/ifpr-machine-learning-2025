import numpy as np
import pandas as pd

def data_set(dataset):
    result = {}

    result['dataset'] = dataset

    # Carregar dataset
    df = pd.read_csv(dataset)
    
    # Definir nomes das colunas se não tiver header
    if df.columns[0] == '0' or pd.isna(df.columns[0]):
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
        ]
        df.columns = column_names
    
    # Remover linhas com valores '?'
    df = df.replace(' ?', np.nan)
    df = df.dropna()
    
    # Obter nome da classe (última coluna)
    cls_name = df.columns[-1]
    
    # Processar classes
    cls_original, cls_original_num, cls_counts = np.unique(df[cls_name], return_inverse=True, return_counts=True)
    
    # Separar features da classe
    features_df = df.drop(columns=[cls_name]).copy()
    
    # Converter colunas categóricas para números usando pandas
    categorical_mappings = {}
    for col in features_df.select_dtypes(include=['object']).columns:
        # Obter valores únicos
        unique_values = features_df[col].unique()
        
        # Criar mapeamento: valor -> número
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        categorical_mappings[col] = mapping
        
        # Aplicar o mapeamento usando pandas map()
        features_df[col] = features_df[col].map(mapping)
    
    # Preencher resultado
    result['data'] = features_df
    result['class'] = cls_original_num
    result['class_name'] = cls_original
    result['class_counts'] = cls_counts
    result['categorical_mappings'] = categorical_mappings
    
    return result

def save_processed_dataset(data, output_path=None):
    """Salva o dataset processado"""
    
    if output_path is None:
        output_path = r'C:\Codes\Machine Learning\2-Tri\adultDataset\manipulated.csv'
    
    # Recriar dataset completo
    complete_df = data['data'].copy()
    complete_df['class'] = data['class']
    
    # Salvar
    complete_df.to_csv(output_path, index=False)
    print(f"Dataset processado salvo em: {output_path}")
    
    # Salvar também o mapeamento das categorias
    mapping_path = output_path.replace('.csv', '_mappings.txt')
    with open(mapping_path, 'w') as f:
        f.write("Categorical Mappings:\n")
        f.write("="*30 + "\n\n")
        for col, mapping in data['categorical_mappings'].items():
            f.write(f"{col}:\n")
            for original, encoded in mapping.items():
                f.write(f"  {original} -> {encoded}\n")
            f.write("\n")
    
    print(f"Mapeamentos salvos em: {mapping_path}")
    
    return output_path

DATASET = r'C:\Codes\Machine Learning\2-Tri\adultDataset\adult.csv'

if __name__ == '__main__':
    data = data_set(DATASET)
    
    for key, value in data.items():
        if key != 'categorical_mappings':  
            print(f'{key} : \n {value}\n')
    
    # Mostrar os mapeamentos criados
    print("Categorical mappings:")
    for col, mapping in data['categorical_mappings'].items():
        print(f"{col}: {mapping}")
        print()
    
    cont = len(data['class_name'])
    print(f"Number of classes: {cont}")
    
    total = np.sum(data['class_counts'])
    ir = []
    for value in data['class_counts']:
        ir.append(total / value)
    
    max_ir = np.max(ir)
    print(f"Imbalance Ratio: {max_ir}")
    
    # Informações adicionais sobre o dataset
    print(f"\nDataset shape: {data['data'].shape}")
    print(f"Classes: {data['class_name']}")
    print(f"Class distribution: {data['class_counts']}")
    
    # Verificar se todos os dados são numéricos
    print(f"\nData types after conversion:")
    print(data['data'].dtypes)
    
    # Salvar manualmente
    save_processed_dataset(data)