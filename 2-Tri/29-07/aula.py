

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
    # todo: salvar adult--classes.csv  
    # ok: remover dados faltantes

    # todo: fazer o split do dataset: treino / teste

