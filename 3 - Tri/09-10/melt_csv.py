import pandas as pd
import numpy as np

def fazer_melt_csv(arquivo_entrada='all.csv', arquivo_saida='all_melted.csv'):

    print(" FAZENDO MELT DO CSV")
    print("="*50)
    
    # Carregar dados
    print(f" Carregando {arquivo_entrada}...")
    df = pd.read_csv(arquivo_entrada)
    print(f"   Formato original: {df.shape[0]} linhas × {df.shape[1]} colunas")
    
    # Identificar colunas de valores (v1-v20)
    value_cols = [f'v{i}' for i in range(1, 21)]
    id_cols = ['dataset', 'classifier', 'metric', 'author']
    
    print(f"   Colunas de ID: {id_cols}")
    print(f"   Colunas de valores: {len(value_cols)} colunas (v1-v20)")
    
    # Fazer o melt
    print(" Aplicando melt...")
    df_melted = pd.melt(
        df, 
        id_vars=id_cols,
        value_vars=value_cols,
        var_name='experimento',
        value_name='valor'
    )
    
    # Converter valores para numérico (importante!)
    print(" Convertendo valores para numérico...")
    df_melted['valor'] = pd.to_numeric(df_melted['valor'], errors='coerce')
    
    # Remover linhas com valores inválidos
    linhas_antes = len(df_melted)
    df_melted = df_melted.dropna(subset=['valor'])
    linhas_depois = len(df_melted)
    
    if linhas_antes != linhas_depois:
        print(f"     Removidas {linhas_antes - linhas_depois} linhas com valores inválidos")
    
    df_melted['experimento_num'] = df_melted['experimento'].str.extract(r'(\d+)').astype(int)
    
    df_melted = df_melted[['dataset', 'classifier', 'metric', 'experimento', 'experimento_num', 'valor', 'author']]
    
    df_melted = df_melted.sort_values(['dataset', 'classifier', 'metric', 'experimento_num'])
    
    print(f"   Formato final: {df_melted.shape[0]} linhas × {df_melted.shape[1]} colunas")
    
    print(f" Salvando {arquivo_saida}...")
    df_melted.to_csv(arquivo_saida, index=False)
    
    # Mostrar estatísticas
    print("\n ESTATÍSTICAS DO MELT:")
    print(f"   Datasets únicos: {df_melted['dataset'].nunique()}")
    print(f"   Classificadores únicos: {df_melted['classifier'].nunique()}")
    print(f"   Métricas únicas: {df_melted['metric'].nunique()}")
    print(f"   Autores únicos: {df_melted['author'].nunique()}")
    print(f"   Experimentos por combinação: {df_melted['experimento'].nunique()}")
    
    # Mostrar alguns exemplos
    print("\n PRIMEIRAS 10 LINHAS DO RESULTADO:")
    print(df_melted.head(10).to_string(index=False))
    
    # Verificar valores faltantes
    valores_nan = df_melted['valor'].isna().sum()
    if valores_nan > 0:
        print(f"\n  ATENÇÃO: {valores_nan} valores faltantes encontrados!")
        # Mostrar quais combinações têm valores faltantes
        nan_data = df_melted[df_melted['valor'].isna()]
        print("   Combinações com valores faltantes:")
        for _, row in nan_data.iterrows():
            print(f"     {row['dataset']} | {row['classifier']} | {row['metric']} | {row['experimento']} | {row['author']}")
    else:
        print("\n Nenhum valor faltante encontrado!")
    
    print(f"\n🎯 MELT CONCLUÍDO!")
    print(f"   Arquivo salvo: {arquivo_saida}")
    
    return df_melted

def criar_filtro_naive_bayes_f1(df_melted):
    """
    Cria um CSV filtrado apenas com Naive Bayes + F1 Score
    """
    print(f"\n📋 CRIANDO FILTRO: NAIVE BAYES + F1 SCORE")
    print("="*50)
    
    # Filtrar apenas Naive Bayes + F1 Score
    filtro = (df_melted['classifier'] == 'naive_bayes') & (df_melted['metric'] == 'f1_score')
    df_nb_f1 = df_melted[filtro].copy()
    
    print(f"   📊 Dados filtrados: {len(df_nb_f1)} registros")
    print(f"   📈 Datasets únicos: {df_nb_f1['dataset'].nunique()}")
    print(f"   👥 Autores únicos: {df_nb_f1['author'].nunique()}")
    
    # Salvar CSV filtrado
    output_file = 'naive_bayes_f1_score.csv'
    df_nb_f1.to_csv(output_file, index=False)
    
    print(f"\n📊 ESTATÍSTICAS DO FILTRO:")
    print(f"   Datasets: {sorted(df_nb_f1['dataset'].unique())}")
    print(f"   Total de experimentos: {len(df_nb_f1)}")
    print(f"   Média geral F1-Score: {df_nb_f1['valor'].mean():.4f}")
    print(f"   Desvio padrão: {df_nb_f1['valor'].std():.4f}")
    print(f"   Range: {df_nb_f1['valor'].min():.4f} - {df_nb_f1['valor'].max():.4f}")
    
    # Estatísticas por dataset
    print(f"\n📈 ESTATÍSTICAS POR DATASET:")
    stats_por_dataset = df_nb_f1.groupby('dataset')['valor'].agg(['count', 'mean', 'std', 'min', 'max'])
    stats_por_dataset.columns = ['Experimentos', 'Média', 'Desvio', 'Mínimo', 'Máximo']
    print(stats_por_dataset.round(4))
    
    # Mostrar algumas linhas de exemplo
    print(f"\n🔍 PRIMEIRAS 10 LINHAS DO FILTRO:")
    colunas_exibir = ['dataset', 'experimento', 'valor', 'author']
    print(df_nb_f1[colunas_exibir].head(10).to_string(index=False))
    
    print(f"\n✅ FILTRO SALVO: {output_file}")
    
    return df_nb_f1

def exemplo_analise(df_melted):
    """
    Exemplo de análise com os dados em formato long
    """
    print("\n EXEMPLO DE ANÁLISE COM DADOS MELTED:")
    print("="*50)
    
    # Filtrar apenas Naive Bayes + F1 Score
    nb_f1 = df_melted[
        (df_melted['classifier'] == 'naive_bayes') & 
        (df_melted['metric'] == 'f1_score')
    ].copy()
    
    if len(nb_f1) > 0:
        print(f" Naive Bayes + F1 Score: {len(nb_f1)} registros")
        
        # Verificar se todos os valores são numéricos
        if nb_f1['valor'].dtype in ['object', 'string']:
            print(" Convertendo valores para numérico...")
            nb_f1['valor'] = pd.to_numeric(nb_f1['valor'], errors='coerce')
            nb_f1 = nb_f1.dropna(subset=['valor'])
            print(f" Após conversão: {len(nb_f1)} registros válidos")
        
        if len(nb_f1) > 0:
            # Estatísticas por dataset
            stats_por_dataset = nb_f1.groupby('dataset')['valor'].agg(['count', 'mean', 'std', 'min', 'max'])
            print("\n Estatísticas por Dataset:")
            print(stats_por_dataset.round(4))
            
            # Melhor e pior dataset
            melhor_dataset = stats_por_dataset['mean'].idxmax()
            pior_dataset = stats_por_dataset['mean'].idxmin()
            
            print(f"\n Melhor dataset: {melhor_dataset} (média: {stats_por_dataset.loc[melhor_dataset, 'mean']:.4f})")
            print(f" Pior dataset: {pior_dataset} (média: {stats_por_dataset.loc[pior_dataset, 'mean']:.4f})")
        else:
            print(" Nenhum valor numérico válido encontrado")
        
    else:
        print(" Nenhum dado de Naive Bayes + F1 Score encontrado")

if __name__ == "__main__":
    # Fazer o melt
    df_melted = fazer_melt_csv()
    
    # Exemplo de análise
    exemplo_analise(df_melted)
    
    # Criar CSV filtrado para Naive Bayes + F1 Score
    criar_filtro_naive_bayes_f1(df_melted)
    
    print("\n SCRIPT CONCLUÍDO!")
    print("   Agora você pode usar 'all_melted.csv' para análises mais fáceis!")
    print("   Criamos também 'naive_bayes_f1_filtered.csv' para análise específica do Naive Bayes!")