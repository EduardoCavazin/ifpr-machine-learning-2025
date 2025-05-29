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
    
#Transformar tabela em objeto array
table = [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
print("Tabela original:", table)
table2 = np.array(table)
print("Tabela transformada em array:", table2)
print("Num. de dimensões:", table2.ndim)
print("Mostar o num. 5:", table2[1,1])
print("Mostrar a última linha:", table2[:-1])
print("Mostrar a primeira coluna:", table2[:,0])

#Exercício 12 -> Transpor a tabela e armazenar em outra variável
tabela = np.array([['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                   ['k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'],
                   ['u', 'v', 'w', 'x', 'y', 'z', '@', '#', '*', '+']])

print('Tabela original:', tabela)
tabela_t = tabela.T
print('Tabela transposta:', tabela_t)

#Exercício 13 -> Capturar o elemento linha=2 e coluna=3
element = tabela[2][3] 
print("Exercício 13:", element)

#Exercício 14 -> Transformar a tabela em um shape (10, 3)
tabela2 = tabela.T
for linha in tabela2:
    print("linha -->", linha)

#Exercício 15 -> Imprimir cada coluna da tabela2
for coluna in tabela2.T:
    print("Coluna -->", coluna)

    
#Exercício 16 -> Capturar da tabela os elementos do meio e colocar na variável: tabela3
tabela3 = tabela.reshape(-1, 5)
tabela3 = tabela3[1:-1, 1:-1]
print("Exercício 16 ->")
print(tabela3)

#Exercício 17 -> Imprimir o shape da tabela3
print("Exercício 17 -> Shape da Tabela3:", tabela3.shape)

#Exercício 18 -> Imprimir todas colunas da tabela3
for i in range(tabela3.shape[1]):
    print("Exercício 18:", tabela3[:, i])
    
#Exercício 19 -> Transformar a tabela 3 em uma lista, e colocar dentro da variável: lista3
lista3 = tabela3.flatten()
print("Exercício 19 -> Lista3:", lista3)

#Exercício 20 -> imprimir na tela, da lista3, os elementos de índice: 1, 4, 7 e 8
elementos_indices = [1, 4, 7, 8]
for i in elementos_indices:
    print(f"Exercício 20 -> Elemento de índice {i}:", lista3[i])
    
lista3 = [item for sublist in lista3 for item in sublist]