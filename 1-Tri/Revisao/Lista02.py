import re
import pandas as pd

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

df = pd.read_csv(r'C:\Codes\Machine Learning\Assets\cpf_placa.csv')

detran_dados = dict(zip(df['CPF'].astype(str), df['PLACA']))

print("="*70)
print("SISTEMA DETRAN - VERIFICAÇÃO DE PLACAS (DATASET REAL)")
print("="*70)
print(f"Total de registros carregados: {len(detran_dados)}")
print("="*70)

print("\nPrimeiros 10 registros:")
print("-"*50)
contador = 0
for cpf, placa in detran_dados.items():
    if contador < 10:
        tipo_placa = verificar_placa(placa)
        print(f"{placa.upper()} -- {cpf} -- {tipo_placa}")
        contador += 1

print("="*70)

antigas = len([placa for cpf, placa in detran_dados.items() if verificar_placa(placa) == "placa antiga"])
novas = len([placa for cpf, placa in detran_dados.items() if verificar_placa(placa) == "placa nova"])
invalidas = len([placa for cpf, placa in detran_dados.items() if verificar_placa(placa) == "formato inválido"])

print("\nESTATÍSTICAS COMPLETAS:")
print("="*40)
print(f"Placas antigas: {antigas}")
print(f"Placas novas: {novas}")
print(f"Placas inválidas: {invalidas}")
print(f"Total de registros: {len(detran_dados)}")
print(f"Percentual antigas: {(antigas/len(detran_dados)*100):.1f}%")
print(f"Percentual novas: {(novas/len(detran_dados)*100):.1f}%")
print(f"Percentual inválidas: {(invalidas/len(detran_dados)*100):.1f}%")

print("\n" + "="*50)
print("ANÁLISE DETALHADA")
print("="*50)

print("\nAMOSTRA DE PLACAS ANTIGAS (5 primeiras):")
contador_antigas = 0
for cpf, placa in detran_dados.items():
    if verificar_placa(placa) == "placa antiga" and contador_antigas < 5:
        print(f"  {placa.upper()} - CPF: {cpf}")
        contador_antigas += 1

print("\nAMOSTRA DE PLACAS NOVAS (5 primeiras):")
contador_novas = 0
for cpf, placa in detran_dados.items():
    if verificar_placa(placa) == "placa nova" and contador_novas < 5:
        print(f"  {placa.upper()} - CPF: {cpf}")
        contador_novas += 1

if invalidas > 0:
    print(f"\nPLACAS INVÁLIDAS ({invalidas} encontradas):")
    for cpf, placa in detran_dados.items():
        if verificar_placa(placa) == "formato inválido":
            print(f"  {placa.upper()} - CPF: {cpf}")

print("\n" + "="*70)
print("ANÁLISE CONCLUÍDA")
print("="*70)