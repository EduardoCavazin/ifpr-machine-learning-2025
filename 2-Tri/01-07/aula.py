import pandas as pd
from dataset import data_set


FNAME = 'datasets/adult/adult.csv'

if __name__ == '__main__':
    data = data_set(FNAME)
    for key, value in data.items():
        print(key)

    fname = FNAME.split('/')
    fname = fname[-1]
    print('fname -->', fname)
    
    # todo: salvar adult--dados.csv
    
    dados_df = data['dados']
    dados_filename = fname.replace('.csv', '--dados.csv')
    dados_df.to_csv(dados_filename, index=False)
    print('dados_df -->', dados_df.shape)    

    # todo: salvar adult--classes.csv  
    
    classes_df = pd.DataFrame({'class': data['classes']})
    classes_filename = fname.replace('.csv', '--classes.csv')
    classes_df.to_csv(classes_filename, index=False)
    print('classes_df -->', classes_df.shape)
    
    # todo: remover números inválidos
    
    