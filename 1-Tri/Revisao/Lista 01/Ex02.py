import keyboard

general_list = []

print("Digite os números (Pressione 'Esc'para sair): ")

while True:
    try:
        if keyboard.is_pressed('esc'):
            print("\nESC pressionado. Parando...")
            break
            
        num = int(input("Digite um número: "))
        general_list.append(num)
    except ValueError:
        print("Entrada inválida. Por favor, digite um número inteiro.")
        continue
    except KeyboardInterrupt:
        print("\nInterrupção do teclado detectada. Parando...")
        break

positive_list = []
negative_list = []
positive_list_sum = 0
negative_list_sum = 0
positive_list_count = 0
negative_list_count = 0

for num in general_list:
    if num > 0:
        positive_list.append(num)
        positive_list_sum += num
        positive_list_count += 1
    elif num < 0:
        negative_list.append(num)
        negative_list_sum += num
        negative_list_count += 1

positive_mean = positive_list_sum / positive_list_count if positive_list_count > 0 else 0
negative_mean = negative_list_sum / negative_list_count if negative_list_count > 0 else 0

print("\n" + "=" * 30)
print("Resultados:")
print("="*30)

print("Números positivos:")
print("Soma:", positive_list_sum)
print("Média:", positive_mean)

print("\nNúmeros negativos:")
print("Soma:", negative_list_sum)
print("Média:", negative_mean)
print("\nNúmeros digitados:", general_list)
print("Números positivos:", positive_list)
print("Números negativos:", negative_list)