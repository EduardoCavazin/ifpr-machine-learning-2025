
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def _transform_col( data ):
    vlr_orig, values, count = np.unique(data, return_inverse=True, return_counts=True)
    result = {}
    result['vlr-orig'] = list(vlr_orig)
    result['values'] = list(values)
    result['vlr-count'] = list(count)
    return result


def _transform_data(data, col_list):
    for colname in list(data.columns):
        if colname not in col_list: continue
        dados = data[ colname ]
        ret = _transform_col( dados )
        ret['colname'] = colname
        data.drop( columns=colname )
        data[ colname ] = ret['values']
    return data


def _ensure_all_numeric(data):
    """
    Garante que todas as colunas sejam numéricas.
    Converte colunas categóricas restantes usando LabelEncoder.
    """
    le = LabelEncoder()
    
    for column in data.columns:
        if data[column].dtype == 'object':
            print(f"Convertendo coluna categórica restante: {column}")
            # Converter NaN para string antes de aplicar LabelEncoder
            data[column] = data[column].astype(str)
            data[column] = le.fit_transform(data[column])
            
    return data


def _convert_classes_to_numeric(classes):
    """
    Converte as classes para valores numéricos.
    Retorna as classes numéricas e o mapeamento.
    """
    le = LabelEncoder()
    classes_numeric = le.fit_transform(classes)
    
    print(f"Classes convertidas: {le.classes_} -> {np.unique(classes_numeric)}")
    
    return classes_numeric, le.classes_



def dataset_info(data):
    ###################
    data.info(verbose=True)
    print(data.describe())
    print('tipos:', data.dtypes)
    print('dimensoes:', data.ndim)
    print('linhas x colunas:', data.shape)
    ###################

def remover_dados_faltantes( df ):
    mascara = df.apply(lambda linha: linha.astype(str).str.contains(r'\?')).any(axis=1)

    # Retorna um DataFrame apenas com as linhas que **não** contêm '?'
    data = df[~mascara].copy()
    return data


def data_set( fname ):
    result = {}
    result['nome-arquivo'] = fname
    data = pd.read_csv(fname, skipinitialspace=True, skip_blank_lines=True)
    
    print("=== INFORMAÇÕES DO DATASET ORIGINAL ===")
    dataset_info(data)

    # Remover dados faltantes
    data = remover_dados_faltantes( data )
    
    print("\n=== APÓS REMOÇÃO DE DADOS FALTANTES ===")
    dataset_info(data)

    # Transformar colunas específicas do Adult dataset
    mystr = 'workclass, education, marital-status, occupation, relationship, race, sex, native-country, class'
    process = [x.strip() for x in mystr.split(',')]
    data = _transform_data(data, process)

    # Separar features e classes
    ultima = data.columns[-1]
    classes = list(data[ultima])
    df = data.drop( columns=ultima )
    
    # Garantir que todos os dados sejam numéricos
    df = _ensure_all_numeric(df)
    
    # Converter classes para valores numéricos também
    classes_numeric, class_mapping = _convert_classes_to_numeric(classes)
    
    print("\n=== DADOS FINAIS PROCESSADOS ===")
    print(f"Shape dos dados: {df.shape}")
    print(f"Tipos de dados:")
    print(df.dtypes)
    print(f"Classes originais: {np.unique(classes)}")
    print(f"Classes numéricas: {np.unique(classes_numeric)}")
    print("Todas as colunas são numéricas:", df.select_dtypes(include=[np.number]).shape[1] == df.shape[1])

    result['dados'] = df
    result['classes'] = classes_numeric  # Retorna classes numéricas
    result['classes_originais'] = classes  # Mantém classes originais para referência
    result['class_mapping'] = class_mapping  # Mapeamento das classes

    return result



