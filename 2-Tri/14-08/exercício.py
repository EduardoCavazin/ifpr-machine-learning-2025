"""
===============================================================================
EXERCÍCIO: CLASSIFICAÇÃO COM PERCEPTRON E SVM
===============================================================================
Requisitos:
- Remover a função capturar_dados
- Chamar a função data_set da aula 01-julho  
- Com os dados do dataset 'adult', fazer:
  - Misturar o dataset com shuffle
  - Separar em treino (60%) e teste (40%)
  - Rodar Perceptron pelo menos 20x
  - Capturar acurácia de cada rodada
  - Calcular a média

Exercício adicional:
- Fazer o mesmo para SVM (Support Vector Machine)
===============================================================================
"""

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn import metrics
from random import shuffle
import numpy as np
import pandas as pd
import sys
import os
import time

sys.path.append('../')
from adultManipulation import data_set


def executar_classificador_multiplas_vezes(classificador_nome, X_treino, y_treino, X_teste, y_teste, num_execucoes=20):
    """
    Executa um classificador múltiplas vezes e calcula estatísticas
    
    Args:
        classificador_nome: "Perceptron" ou "SVM"
        X_treino, y_treino: dados de treino
        X_teste, y_teste: dados de teste  
        num_execucoes: número de execuções (default: 20)
    
    Returns:
        lista de acurácias, média das acurácias
    """
    acuracias = []
    tempos = []
    
    print(f"\n=== {classificador_nome} - {num_execucoes} execuções ===")
    print("Execução | Acurácia | Tempo (s)")
    print("---------|----------|----------")
    
    for i in range(num_execucoes):
        # Marcar início do tempo
        inicio_tempo = time.time()
        
        # Criar novo modelo para cada execução (com random_state diferente)
        # if classificador_nome == "Perceptron":
        #     modelo = Perceptron(max_iter=1000, random_state=i)
        # elif classificador_nome == "SVM":
        #     modelo = SVC(random_state=i, gamma='auto', kernel='rbf')
        
        modelo = Perceptron(max_iter=1000, random_state=i)
        
        # Treinar o modelo
        modelo.fit(X_treino, y_treino)
        
        # Fazer predições
        y_pred = modelo.predict(X_teste)
        
        # Marcar fim do tempo
        fim_tempo = time.time()
        tempo_execucao = fim_tempo - inicio_tempo
        
        # Calcular acurácia
        acuracia = metrics.accuracy_score(y_teste, y_pred)
        acuracias.append(acuracia)
        tempos.append(tempo_execucao)
        
        print(f"   {i+1:2d}    | {acuracia:.4f}   | {tempo_execucao:.3f}")
    
    # Calcular estatísticas de acurácia
    media = np.mean(acuracias)
    desvio = np.std(acuracias)
    minimo = np.min(acuracias)
    maximo = np.max(acuracias)
    
    # Calcular estatísticas de tempo
    tempo_medio = np.mean(tempos)
    tempo_total = np.sum(tempos)
    tempo_min = np.min(tempos)
    tempo_max = np.max(tempos)
    
    print(f"\n--- Estatísticas {classificador_nome} ---")
    print(f"📊 ACURÁCIA:")
    print(f"   Média:         {media:.4f}")
    print(f"   Desvio padrão: {desvio:.4f}")
    print(f"   Mínimo:        {minimo:.4f}")
    print(f"   Máximo:        {maximo:.4f}")
    print(f"⏱️  TEMPO:")
    print(f"   Tempo médio:   {tempo_medio:.3f}s")
    print(f"   Tempo total:   {tempo_total:.3f}s")
    print(f"   Mais rápido:   {tempo_min:.3f}s")
    print(f"   Mais lento:    {tempo_max:.3f}s")
    
    return acuracias, media


def mostrar_matriz_confusao(y_real, y_pred, nome_modelo):
    """
    Mostra matriz de confusão e métricas detalhadas
    """
    print(f"\n=== MATRIZ DE CONFUSÃO - {nome_modelo} ===")
    
    matriz = metrics.confusion_matrix(y_real, y_pred)
    print("Matriz de Confusão:")
    print(matriz)
    
    # Extrair componentes da matriz (para classificação binária)
    tn, fp, fn, tp = matriz.ravel()
    
    print(f"\nComponentes da Matriz:")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")
    
    # Calcular métricas
    acuracia = (tp + tn) / (tp + tn + fp + fn)
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
    
    print(f"\nMétricas:")
    print(f"Acurácia:  {acuracia:.4f}")
    print(f"Precisão:  {precisao:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")


def main():
    print("EXERCÍCIO: CLASSIFICAÇÃO COM PERCEPTRON E SVM ")
    print("=" * 70)
    
    print("\n 1. Carregando dataset Adult")
    try:
        adult_path = '../adultDataset/adult.csv'
        data = data_set(adult_path)
        
        
        xdata = np.array(data['data'])  
        ytarg = np.array(data['class']) 
        
        print(f"Dataset carregado com sucesso!")
        print(f"   Amostras: {xdata.shape[0]}")
        print(f"   Features: {xdata.shape[1]}")
        print(f"   Classes únicas: {np.unique(ytarg)}")
        print(f"   Classes originais: {data['class_name']}")
        print(f"   Distribuição: {data['class_counts']}")
        
        
    except Exception as e:
        print(f"Erro ao carregar dataset: {e}")
        return
    
    
    print("\n2. Embaralhando os dados...")
    indices = list(range(len(ytarg)))
    shuffle(indices)
    
    xdata = xdata[indices]
    ytarg = ytarg[indices]
    print("Dados embaralhados com sucesso!")
    
    print("\n3. Dividindo dataset...")
    size = len(ytarg)
    particao_treino = int(size * 0.6)
    
    X_treino = xdata[:particao_treino]
    y_treino = ytarg[:particao_treino]
    X_teste = xdata[particao_treino:]
    y_teste = ytarg[particao_treino:]
    
    print(f"Treino: {len(X_treino)} amostras (60%)")
    print(f"Teste:  {len(X_teste)} amostras (40%)")

    print(f"\n4. EXPERIMENTO")
    
    
    X_treino_final = X_treino
    X_teste_final = X_teste
    
    
    print("\n" + "="*60)
    print("PERCEPTRON - 20 EXECUÇÕES")
    print("="*60)
    
    acuracias_perceptron, media_perceptron = executar_classificador_multiplas_vezes(
        "Perceptron", X_treino_final, y_treino, X_teste_final, y_teste, 20
    )
    
    print("\n" + "="*60)
    print("SVM (SUPPORT VECTOR MACHINE) - 20 EXECUÇÕES")
    print("="*60)
    
    acuracias_svm, media_svm = executar_classificador_multiplas_vezes(
        "SVM", X_treino_final, y_treino, X_teste_final, y_teste, 20
    )
    
    print("\n" + "="*60)
    print("COMPARAÇÃO FINAL")
    print("="*60)
    print(f"Perceptron - Média: {media_perceptron:.4f}")
    print(f"SVM       - Média: {media_svm:.4f}")
    
    diferenca = abs(media_svm - media_perceptron)
    if media_svm > media_perceptron:
        print(f"SVM foi melhor por {diferenca:.4f} pontos!")
    elif media_perceptron > media_svm:
        print(f"Perceptron foi melhor por {diferenca:.4f} pontos!")
    else:
        print("Empate técnico!")
    
    
    melhor_modelo = "SVM" if media_svm >= media_perceptron else "Perceptron"
    print(f"\nGerando matriz de confusão do melhor modelo ({melhor_modelo})...")
    
    if melhor_modelo == "SVM":
        modelo_final = SVC(random_state=42, gamma='auto', kernel='rbf')
    else:
        modelo_final = Perceptron(max_iter=1000, random_state=42)
    
    modelo_final.fit(X_treino_final, y_treino)
    y_pred_final = modelo_final.predict(X_teste_final)
    
    mostrar_matriz_confusao(y_teste, y_pred_final, f"{melhor_modelo} (SEM NORMALIZAÇÃO)")
    


if __name__ == "__main__":
    main()
