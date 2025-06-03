import numpy as np
import pandas as pd

FNAME = 'C:\\Codes\\Machine Learning\\Assets\\iris.csv'

def data_set(fname):
    result = {}
    
    result['fName'] = fname
    
    data = pd.read_csv(fname)
    col = data.columns
    
    lastCol = col[-1]
    
    original_name = data[lastCol]
    species = np.unique(original_name)
    original_species, species = np.unique(original_name, return_inverse=True)
    print("Unique species:", species)

    print("-"*40)

    species = data[lastCol]
    df = data.drop(columns=[lastCol])
    print(species)
    print(df)
    result['data'] = df


    return result



if __name__ == '__main__':
    data = data_set(FNAME)
    

    
    
    