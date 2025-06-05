import keyboard as kb

cardapio = {}

cardapio['Dogão'] = 10.00
cardapio['X-Burguer'] = 12.00
cardapio['X-Salada'] = 15.00
cardapio['X-Bacon'] = 18.00
cardapio['X-Tudo'] = 20.00

cardapio['Refrigerante'] = 5.00
cardapio['Suco'] = 6.00
cardapio['Água'] = 3.00

print("="*50)
print("Cardápio:")
print("="*50)

for lanche, preco in cardapio.items():
    print(f"{lanche}: R$ {preco:.2f}")
print("="*50)

conta_total = 0.0
todos_produtos = []

print("\nDigite seus pedidos (pressione ESC a qualquer momento para finalizar):")


while True:
    try:
        # Verifica se ESC foi pressionado
        if kb.is_pressed('esc'):
            print("\n\nESC pressionado. Finalizando pedido...")
            break
            
        print("\n" + "-"*30)
        lanche_pedido = input("Digite o nome do lanche: ")
        
        # Verifica ESC novamente
        if kb.is_pressed('esc'):
            print("\nESC pressionado. Finalizando pedido...")
            break
            
        bebida_pedido = input("Digite o nome da bebida: ")
        
        produtos_pedido_atual = []
        valor_pedido_atual = 0.0

        # Verifica lanche
        if lanche_pedido in cardapio:
            conta_total += cardapio[lanche_pedido]
            valor_pedido_atual += cardapio[lanche_pedido]
            produtos_pedido_atual.append(lanche_pedido)
            todos_produtos.append(lanche_pedido)
            print(f"✓ {lanche_pedido} adicionado - R$ {cardapio[lanche_pedido]:.2f}")
        else:
            print(f"✗ Lanche '{lanche_pedido}' não encontrado no cardápio.")

        # Verifica bebida
        if bebida_pedido in cardapio:
            conta_total += cardapio[bebida_pedido]
            valor_pedido_atual += cardapio[bebida_pedido]
            produtos_pedido_atual.append(bebida_pedido)
            todos_produtos.append(bebida_pedido)
            print(f"✓ {bebida_pedido} adicionado - R$ {cardapio[bebida_pedido]:.2f}")
        else:
            print(f"✗ Bebida '{bebida_pedido}' não encontrada no cardápio.")
        
        # Mostra subtotal do pedido atual
        if produtos_pedido_atual:
            print(f"\nSubtotal deste pedido: R$ {valor_pedido_atual:.2f}")
            print(f"Total acumulado: R$ {conta_total:.2f}")
        
    except KeyboardInterrupt:
        print("\n\nPrograma interrompido.")
        break
    except Exception as e:
        print(f"Erro: {e}")
        continue

# Resumo final
print("\n" + "="*50)
print("RESUMO FINAL DO PEDIDO")
print("="*50)

if todos_produtos:
    print("Produtos pedidos:")
    for produto in todos_produtos:
        print(f"• {produto}: R$ {cardapio[produto]:.2f}")
    print("-"*50)
    print(f"TOTAL A PAGAR: R$ {conta_total:.2f}")
else:
    print("Nenhum produto foi pedido.")

print("="*50)
