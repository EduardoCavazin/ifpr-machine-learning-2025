from sklearn.linear_model import Perceptron
from sklearn import metrics
from random import shuffle
import numpy as np


def capturar_dados():
    data = {'dados': None, 'classes': None}
    data['dados'] = [ 
                        [1 ,2 ,3 , 4],
                        [11,12,13,14],
                        [21,22,23,24],
                        [31,32,33,34],
                        [41,42,43,44],
                        [51,52,53,54],
                        [61,62,63,64],
                        [71,72,73,74],
                        [81,82,83,84],
                        [91,92,93,94],
                    ]
    data['classes'] = [0, 1, 0, 1, 1, 1, 0, 0, 0, 1]
    return data


data = capturar_dados()
xdata = data['dados']
ytarg = data['classes']

xdata = np.array( xdata ) # np.array -> para embaralhar dados
ytarg = np.array( ytarg )

# embaralhar os dados
nums = list(range(len(ytarg)))
print(nums)
shuffle(nums)
print(nums)

xdata = xdata[ nums ]
ytarg = ytarg[ nums ]

size = len(ytarg)
particao = int(size*0.5) # treino -> 50%

xtreino = xdata[ : particao ]
ytreino = ytarg[ : particao ]

xteste = xdata[ particao : ]
yteste = ytarg[ particao : ]


print(xtreino)
print(ytreino)

print(xteste)
print(yteste)


perceptron = Perceptron(max_iter=100,random_state=42)
perceptron.fit(xtreino, ytreino)
yhat = perceptron.predict( xteste )

score = metrics.accuracy_score( yteste, yhat )
matrix = metrics.confusion_matrix( yteste, yhat )
# https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/

print('Evaluating DS techniques:')
print('perceptron-score:', score)
print('confusion-matrix:\n', matrix)

