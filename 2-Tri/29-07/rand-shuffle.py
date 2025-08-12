import pandas as pd
import numpy as np
import random


def test1():
    arr_1d = np.array([10, 20, 30, 40, 50])
    print("Before shuffle (1D):", arr_1d)
    np.random.shuffle(arr_1d)
    print("After shuffle (1D):", arr_1d)
    
#test1()


def test2():
    ll = list(range(0,10))
    print(ll)
    random.shuffle(ll)
    print(ll)

#test2()


def test_shuffle():
    fname = 'aaa.csv'
    data = pd.read_csv(fname, header=None)
    ll = list(range(len(data)))
    random.shuffle(ll)

    print(data)
    print(data.iloc[ ll ])

#test_shuffle()



def test_shuffle_v2():
    fname = 'aaa.csv'
    data = pd.read_csv(fname, header=None)
    ll = list(range(len(data)))
    random.shuffle(ll)
    print(data)
    print(data.iloc[ ll ])
    mydata = data.values.tolist()
    #print(mydata)
    for idx in ll:
        print('idx ->', mydata[idx])

#test_shuffle_v2()


def test_mascara():
    fname = 'aaa.csv'
    data = pd.read_csv(fname, header=None)
    mascara = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    mascara = np.array( mascara, dtype=bool )

    mascara[7] = True
    mascara[15] = True
    mascara[19] = True

    mascara[ [7,15,19] ] = True

    df = data[ mascara ]
    print(df)

