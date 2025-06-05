frase = input("Digite uma frase: ")

vogais = "aeiouAEIOU"

contador_vogal = 0

for letra in frase:
    if letra in vogais:
        contador_vogal += 1

print("Número de vogais:", contador_vogal)