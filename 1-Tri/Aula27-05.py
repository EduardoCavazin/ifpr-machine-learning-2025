import numpy as np

lista = [0,1,2,3,4,5,6,7,8,9]
lista = np.array(lista)

print(lista)
lista1 = lista.reshape(2, -1)
print('Lista reshape 1:', lista1)
lista2 = lista.reshape(-1, 2)
print('Lista reshape 2:', lista2)