import re

def verificar_placa(placa):
    
    placa = placa.upper()
    
    padrao_antigo = r'^[A-Z]{3}\d{4}$'
    
    padrao_novo = r'^[A-Z]{3}\d[A-Z]\d{2}$'
    
    if re.match(padrao_antigo, placa):
        return "placa antiga"
    elif re.match(padrao_novo, placa):
        return "placa nova"
    else:
        return "formato inválido"

placas = ["AAA0001", "bbb0002", "CCC0A00", "ddd1B11", "eee2233", "fff9999", "ggg0C22"]

print("="*50)
print("VERIFICAÇÃO DE PLACAS DE TRÂNSITO (REGEX)")
print("="*50)

for placa in placas:
    tipo_placa = verificar_placa(placa)
    print(f"{placa.upper()} -- {tipo_placa}")

print("="*50)

antigas = len([p for p in placas if verificar_placa(p) == "placa antiga"])
novas = len([p for p in placas if verificar_placa(p) == "placa nova"])
invalidas = len([p for p in placas if verificar_placa(p) == "formato inválido"])

print("\nEstatísticas:")
print(f"Placas antigas: {antigas}")
print(f"Placas novas: {novas}")
print(f"Placas inválidas: {invalidas}")
print(f"Total de placas: {len(placas)}")