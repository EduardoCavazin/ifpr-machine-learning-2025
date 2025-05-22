# INTRO

# criando uma string com várias letras, para nossos testes..
letras = 'abcdefghijklmnopqrstuvwxyz@#*+'

# criando uma lista de letras
lista = list(letras)
print('len(lista):', len(lista))
print(lista)

# importando a biblioteca matemática: numpy
import numpy as np

# transformando nossa lista em um array do numpy
lista = np.array(lista)


print('tamanho:', len(lista))
print('shape:', lista.shape)
print('ndim:', lista.ndim)
print('lista:', lista)
print('lista-reverso:', lista[::-1])



# dado a lista anterior, faça os exercícios:

# 1- capturar os primeiros 10 elementos e imprimir na tela
# 2- capturar os últimos 10 elementos e imprimir na tela
# 3- capturar os 10 elementos do meio e imprimir na tela
# 4- imprimir o 21o elemento apenas
# 5- imprimir todos elementos, menos os 5 últimos
# 6- imprimir todos elementos do início até o meio
# 7- imprimir todos elementos do meio até o final
# 8- imprimir todos elementos a partir do 5 , menos os 5 últimos
# 9- imprimir o 12 elemento
# 10- fazer um laço que repita 10 vezes, imprimindo cada vez 3 elementos
    

tabela = lista.reshape((5,6))
print('shape:', tabela.shape)
print('ndim:', tabela.ndim)

print(tabela)
    
print('-- linhas:')
for linha in tabela:
    print(linha)

print('\n-- colunas:')
for coluna in tabela.T: # aqui, o T significa transpose..
    print(coluna)


# EXERCICIOSSSS....
# 11- verificar o que significa 'TRANSPOSE' na internet

# 12- fazer o transpose da tabela e armazenar em outra variável: tabela_t
#       imprimir a tabela normal e sua transposta
tabela = np.matrix([['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                     ['k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'],
                     ['u', 'v', 'w', 'x', 'y', 'z', '@', '#', '*', '+']])         
#tabela_t = ??

print('Tabela', tabela)
print('Tabela Transposta',tabela_t)

# 13- capturar da tabela o elemento linha=2 e coluna=3, e imprimir na tela

# 14- transformar a tabela em um shape (10, 3), armazenar em tabela2. 
#       Imprimir cada linha da tabela2
#       Comparar o resultado com a pergunta: 
tabela2 = ??
for linha in tabela2:
    print(linha)
print('Quantidade de linhas da tabela 2:', ??)

# 15- imprimir as colunas da tabela2
print('Imprimindo colunas', ??)

#16- capturar da tabela, os elementos do meio, e colocar na variável: tabela3
#       Imprimir a tabela3. Abaixo o que deve aparecer:
#       ['h' 'i' 'j' 'k']
#       ['n' 'o' 'p' 'q']
#       ['t' 'u' 'v' 'w']
tabela3 = tabela[ ?? ]
print('Tabela 3',tabela3)

# 17- imprimir o shape da tabela3
print('Shape',tabela.shape)

# 18- imprimir todas colunas da tabela3
tabela3_t = ??
print('Colunas tabela 3', tabela3_t)

# 19- transformar a tabela 3 em uma lista, e colocar dentro da variável: lista3
#       imprimir a lista3
lista3 = ??
print('Transformando tabela 3 em lista 3:',lista3)

# 20- imprimir na tela, da lista3, os elementos de índice: 1, 4, 7 e 8











# dado a lista anterior, faça os exercícios:

# 1- capturar os primeiros 10 elementos e imprimir na tela
print('Lista dez primeiros :', lista[:10])

# 2- capturar os últimos 10 elementos e imprimir na tela
print('Lista dez ultimos :', lista[-10:])

# 3- capturar os 10 elementos do meio e imprimir na tela

meio = len(lista) // 2 # encontra o índice do meio da lista
print('Lista meio dez',lista[10:-10])

# 4- imprimir o 21o elemento apenas
print('Lista posição 21 :', lista[20])

# 5- imprimir todos elementos, menos os 5 últimos
print('Lista todos menos os 5 ultimos :', lista[:-5])

# 6- imprimir todos elementos do início até o meio
meio = len(lista) // 2 # encontra o meio da lista 
print('Lista até o meio',lista[:meio]) 

# 7- imprimir todos elementos do meio até o final
meio = len(lista) // 2 # encontra o meio da lista 
print('Lista do meio até o final',lista[meio:]) #

# 8- imprimir todos elementos a partir do 5 , menos os 5 últimos
print('Lista a partir dos 5, menos os 5',lista[4:-5])

# 9- imprimir o 12 elemento
print('12 elemento', lista[11])

# 10- fazer um laço que repita 10 vezes, imprimindo cada vez 3 elementos
for rep in range(0,30, 3):
    print(lista[rep:rep+3])
    




# 11- verificar o que significa 'TRANSPOSE' na internet
# transposição é uma maneira de girar uma matriz em 90 graus, trocando as linhas pelas colunas e vice-versa.

# 12- fazer o transpose da tabela e armazenar em outra variável: tabela_t
#       imprimir a tabela normal e sua transposta
tabela = np.matrix([['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                     ['k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'],
                     ['u', 'v', 'w', 'x', 'y', 'z', '@', '#', '*', '+']])         
tabela_t = np.transpose(tabela)

print('Tabela', tabela)
print('Tabela Transposta',tabela_t)

# 13- capturar da tabela o elemento linha=2 e coluna=3, e imprimir na tela
print('Tabela elemento Linha 2 Coluna 3 = ',(tabela.item(1,2)))

# 14- transformar a tabela em um shape (10, 3), armazenar em tabela2. 
#       Imprimir cada linha da tabela2
#       Comparar o resultado com a pergunta: 
tabela2 = tabela.reshape((10,3))
for linha in tabela2:
    print(linha)    
print('Quantidade de linhas da tabela 2:', tabela2.shape[0])

# 15- imprimir as colunas da tabela2
print('Imprimindo colunas', np.transpose(tabela2))

#16- capturar da tabela, os elementos do meio, e colocar na variável: tabela3
#       Imprimir a tabela3. Abaixo o que deve aparecer:
#       ['h' 'i' 'j' 'k']
#       ['n' 'o' 'p' 'q']
#       ['t' 'u' 'v' 'w']
tabela3 = tabela[1:-1, 1:-1]
print('Tabela 3',tabela3)

# 17- imprimir o shape da tabela3
print('Shape',tabela.shape)

# 18- imprimir todas colunas da tabela3
tabela3_t = np.transpose(tabela3)
print('Colunas tabela 3', tabela3_t)

# 19- transformar a tabela 3 em uma lista, e colocar dentro da variável: lista3
#       imprimir a lista3
lista3 = tabela3.tolist()
lista3 = [item for sublist in lista3 for item in sublist] # para transformar a lista em uma lista simples
print('Transformando tabela 3 em lista 3:',lista3)
# 20- imprimir na tela, da lista3, os elementos de índice: 1, 4, 7 e 8
#       OBS: todos estes itens devem ser impressos todos em uma única linha
#print('Imprimindo índices:',lista3[1],lista3[4],lista3[7],lista3[8])



