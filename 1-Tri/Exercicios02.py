import numpy as np

letras = 'abcdefghijklmnopqrstuvwxyz@#*+'

#Criando lista
lista = list(letras)
print('len(lista):', len(lista))
print(lista)


#Transformando lista em array
lista = np.array(lista)
print('tamanho:', len(lista))
print('shape:', lista.shape)
print('ndim:', lista.ndim)
print('lista:', lista)

#Criando tabela
tabela = lista.reshape((5,6))
print('shape:', tabela.shape)
print('ndim:', tabela.ndim)
print(tabela)




#Exercício 01 -> Capturar os 10 primeiros elementos
first_10 = lista[:10]
print("Exercício 01 :" , first_10)

#Exercício 02 -> Capturar os 10 últimos elementos
last_10 = lista[-10:]
print("Exercício 02 :" , last_10)


#Exercício 03 -> Capturar os 10 elementos do meio
middle_10 = lista[10:20]
print("Exercício 03 :" , middle_10)

#Exercício 04 -> Imprimir o 21º elemento
twenty_first = lista[20]
print("Exercício 04 :" , twenty_first)

#Exercício 05 -> Imprimir todos os elementos, menos os 5 últimos
all_but_last_5 = lista[:-5]
print("Exercício 05 :" , all_but_last_5)

#Exercício 06 -> Imprimir todos os elementos do início até o meio
half = len(lista) // 2
start_to_half = lista[:half]
print("Exercício 06 :" , start_to_half)

#Exercício 07 -> Imprimir todos os elementos do meio até o final
middle_to_end = lista[half:]
print("Exercício 07 :" , middle_to_end)

#Exercício 08 -> Imprimir todos os elementos a partir do 5º, menos os 5 últimos
from_5_but_last_5 = lista[4:-5]
print("Exercício 08 :" , from_5_but_last_5)

#Exercício 09 -> Imprimir o 12º elemento
twelfth = lista[11]
print("Exercício 09 :" , twelfth)

#Exercício 10 -> Laço de repetição (10x), imprimindo 3 elementos por repetição
for i in range(0, len(lista), 3):
    print("Exercício 10 :" , lista[i:i+3])