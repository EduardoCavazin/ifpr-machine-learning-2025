import pandas as pd
import numpy as np
import os

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
dataPath = os.path.join(parent_dir, "Assets", "dados.csv")
print(f"Loading data from: {dataPath}")
df = pd.read_csv(dataPath)

# Map numeric classifications to names
df['nome'] = df['classificacao'].map({1: 'João', 2: 'Maria'})

# Count by name
name_counts = df['nome'].value_counts()
print("\nName Counts")
print(name_counts)
print("--"*50)

totalCount = len(df)
print("\nTotal Count")
print(totalCount)
print("--"*50)

# Filter data by name
joao_data = df[df['nome'] == 'João']
maria_data = df[df['nome'] == 'Maria']

#Total distance and descer
total_distance_joao = np.sum(joao_data['distancia'])
total_desceu_joao = np.sum(joao_data['desceu'])
print("\nTotal Distance and Desceu for João")
print(total_distance_joao)
print(total_desceu_joao)
print("--"*50)

total_distance_maria = np.sum(maria_data['distancia'])
total_desceu_maria = np.sum(maria_data['desceu'])
print("\nTotal Distance and Desceu for Maria")
print(total_distance_maria)
print(total_desceu_maria)
print("--"*50)

# Get maximum values
max_distance_joao = np.max(joao_data['distancia'])
print("\nMax Distance for João")
print(max_distance_joao)
print("--"*50)

max_desceu_maria = np.max(maria_data['desceu'])
print("\nMax Desceu for Maria")
print(max_desceu_maria)
print("--"*50)

#Mean values
mean_distance_joao = np.mean(joao_data['distancia'])
mean_desceu_joao = np.mean(joao_data['desceu'])
print("\nMean Distance and Desceu for João")
print(mean_distance_joao)
print(mean_desceu_joao)
print("--"*50)

mean_distance_maria = np.mean(maria_data['distancia'])
mean_desceu_maria = np.mean(maria_data['desceu'])
print("\nMean Distance and Desceu for Maria")
print(mean_distance_maria)
print(mean_desceu_maria)
print("--"*50)

#Total distance and descer
total_distance_joao = np.sum(joao_data['distancia'])
total_desceu_joao = np.sum(joao_data['desceu'])
print("\nTotal Distance and Desceu for João")
print(total_distance_joao)
print(total_desceu_joao)
print("--"*50)
