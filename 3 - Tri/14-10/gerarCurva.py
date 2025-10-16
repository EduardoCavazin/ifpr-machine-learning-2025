import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def carregar_dados_roc(filename):
    """Carrega os dados de ROC do arquivo CSV gerado pelo experimento"""
    print(f"Carregando dados de {filename}...")
    df = pd.read_csv(filename)

    # Organizar dados por classificador
    roc_data = {}

    for clf_name in df['classifier'].unique():
        clf_df = df[df['classifier'] == clf_name]

        y_true = clf_df['y_true'].values
        y_proba_str = clf_df['y_proba'].values

        # Converter strings de probabilidades de volta para arrays
        y_proba = []
        for proba_str in y_proba_str:
            if '|' in str(proba_str):  # Múltiplas classes
                proba = np.array([float(p) for p in str(proba_str).split('|')])
            else:  # Classe única
                proba = float(proba_str)
            y_proba.append(proba)

        roc_data[clf_name] = {
            'y_true': y_true,
            'y_proba': np.array(y_proba)
        }

    print(f"   Dados carregados para {len(roc_data)} classificadores")
    return roc_data


def gerar_curvas_roc(roc_data):
    """Gera curvas ROC para classificação multiclasse usando estratégia One-vs-Rest"""
    print("\nGerando curvas ROC...")

    # Configurar gráfico
    plt.figure(figsize=(12, 8))

    # Cores para cada classificador
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])

    for clf_name, color in zip(roc_data.keys(), colors):
        y_true = roc_data[clf_name]['y_true']
        y_proba = roc_data[clf_name]['y_proba']

        # Verificar se temos probabilidades multiclasse
        if y_proba.ndim == 2:  # Probabilidades para múltiplas classes
            # Binarizar labels
            classes = np.unique(y_true)
            n_classes = len(classes)
            y_true_bin = label_binarize(y_true, classes=classes)

            # Se só temos 2 classes, label_binarize retorna array 1D
            if n_classes == 2:
                y_true_bin = np.hstack([1 - y_true_bin.reshape(-1, 1), y_true_bin.reshape(-1, 1)])

            # Garantir que y_proba tenha o mesmo número de colunas que classes
            if y_proba.shape[1] != n_classes:
                print(f"   AVISO: {clf_name} tem {y_proba.shape[1]} probabilidades, mas {n_classes} classes")
                # Ajustar se necessário
                if y_proba.shape[1] > n_classes:
                    y_proba = y_proba[:, :n_classes]

            # Calcular ROC e AUC para cada classe e fazer macro-average
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Calcular macro-average (média das AUCs de cada classe)
            roc_auc_macro = np.mean([roc_auc[i] for i in range(n_classes)])

            # Para plotar, usar a classe com melhor AUC ou fazer interpolação
            # Vamos usar a média das curvas ROC (macro-average)
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes

            # Plotar curva macro-average
            plt.plot(all_fpr, mean_tpr,
                    color=color, lw=2,
                    label=f'{clf_name} (AUC = {roc_auc_macro:.3f})')

            print(f"   {clf_name}: AUC (macro) = {roc_auc_macro:.3f}")

        else:  # Classificação binária com decision_function
            # Para SVM com decision_function
            y_true_binary = (y_true == np.max(y_true)).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba)
            roc_auc_value = auc(fpr, tpr)

            plt.plot(fpr, tpr,
                    color=color, lw=2,
                    label=f'{clf_name} (AUC = {roc_auc_value:.3f})')

            print(f"   {clf_name}: AUC = {roc_auc_value:.3f}")

    # Plotar linha diagonal (classificador aleatório)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Classificador Aleatório (AUC = 0.500)')

    # Configurar gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=14, fontweight='bold')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=14, fontweight='bold')
    plt.title('Curvas ROC - Dataset Automobile', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    # Salvar gráfico
    plt.tight_layout()
    plt.savefig('curva_roc_automobile.png', dpi=300, bbox_inches='tight')
    print("\nGráfico salvo: curva_roc_automobile.png")

    plt.show()


def carregar_metricas_csv(filename):
    """Carrega F1 Score e Acurácia do arquivo CSV de resultados"""
    print(f"\nCarregando métricas de {filename}...")
    df = pd.read_csv(filename, header=None)

    metricas = {}

    for _, row in df.iterrows():
        dataset = row[0]
        clf_name = row[1]
        metric_name = row[2]
        valores = row[3:].values.astype(float)

        if clf_name not in metricas:
            metricas[clf_name] = {}

        metricas[clf_name][metric_name] = {
            'valores': valores,
            'media': np.mean(valores)
        }

    print(f"   Métricas carregadas para {len(metricas)} classificadores")
    return metricas


def gerar_curvas_roc_com_metricas(roc_data, metricas, metric_type='f1_score'):
    """Gera curvas ROC ponderadas por F1 Score ou Acurácia"""
    metric_name = 'F1 Score' if metric_type == 'f1_score' else 'Acurácia'
    print(f"\nGerando curvas ROC baseadas em {metric_name}...")

    # Configurar gráfico
    plt.figure(figsize=(12, 8))

    # Cores para cada classificador
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])

    for clf_name, color in zip(roc_data.keys(), colors):
        y_true = roc_data[clf_name]['y_true']
        y_proba = roc_data[clf_name]['y_proba']

        # Pegar média da métrica
        media_metrica = metricas[clf_name][metric_type]['media']

        # Verificar se temos probabilidades multiclasse
        if y_proba.ndim == 2:  # Probabilidades para múltiplas classes
            # Binarizar labels
            classes = np.unique(y_true)
            n_classes = len(classes)
            y_true_bin = label_binarize(y_true, classes=classes)

            # Se só temos 2 classes, label_binarize retorna array 1D
            if n_classes == 2:
                y_true_bin = np.hstack([1 - y_true_bin.reshape(-1, 1), y_true_bin.reshape(-1, 1)])

            # Garantir que y_proba tenha o mesmo número de colunas que classes
            if y_proba.shape[1] != n_classes:
                if y_proba.shape[1] > n_classes:
                    y_proba = y_proba[:, :n_classes]

            # Calcular ROC e AUC para cada classe
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Calcular AUC ponderado pela métrica escolhida
            roc_auc_ponderado = np.mean([roc_auc[i] for i in range(n_classes)]) * media_metrica

            # Interpolar curvas ROC
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes

            # Ponderar TPR pela métrica
            mean_tpr_ponderado = mean_tpr * media_metrica

            # Plotar curva
            plt.plot(all_fpr, mean_tpr_ponderado,
                    color=color, lw=2,
                    label=f'{clf_name} (AUC×{metric_name} = {roc_auc_ponderado:.3f}, {metric_name}={media_metrica:.3f})')

            print(f"   {clf_name}: AUC×{metric_name} = {roc_auc_ponderado:.3f} ({metric_name} média = {media_metrica:.3f})")

        else:  # Classificação com decision_function
            y_true_binary = (y_true == np.max(y_true)).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba)
            roc_auc_value = auc(fpr, tpr)

            # Ponderar pelo metric
            roc_auc_ponderado = roc_auc_value * media_metrica
            tpr_ponderado = tpr * media_metrica

            plt.plot(fpr, tpr_ponderado,
                    color=color, lw=2,
                    label=f'{clf_name} (AUC×{metric_name} = {roc_auc_ponderado:.3f}, {metric_name}={media_metrica:.3f})')

            print(f"   {clf_name}: AUC×{metric_name} = {roc_auc_ponderado:.3f} ({metric_name} média = {media_metrica:.3f})")

    # Plotar linha diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Referência')

    # Configurar gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=14, fontweight='bold')
    plt.ylabel(f'TPR ponderado por {metric_name}', fontsize=14, fontweight='bold')
    plt.title(f'Curvas ROC ponderadas por {metric_name} - Dataset Automobile', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Salvar gráfico
    plt.tight_layout()
    filename = f'curva_roc_{metric_type}_automobile.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nGráfico salvo: {filename}")

    plt.show()


def gerar_graficos_barras(metricas):
    """Gera gráficos de barras para F1 Score e Acurácia"""
    print("\nGerando gráficos de barras para F1 Score e Acurácia...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    classificadores = list(metricas.keys())
    cores = ['blue', 'red', 'green', 'orange', 'purple']

    # Extrair médias
    f1_scores = [metricas[clf]['f1_score']['media'] for clf in classificadores]
    accuracies = [metricas[clf]['accuracy']['media'] for clf in classificadores]

    # Gráfico 1: F1 Score
    bars1 = ax1.bar(classificadores, f1_scores, color=cores, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_title('F1 Score Médio por Classificador', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Classificadores', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    # Adicionar valores no topo das barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Gráfico 2: Acurácia
    bars2 = ax2.bar(classificadores, accuracies, color=cores, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_title('Acurácia Média por Classificador', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Classificadores', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Acurácia', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    # Adicionar valores no topo das barras
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig('comparacao_metricas_barras.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo: comparacao_metricas_barras.png")

    plt.show()


def main():
    print("="*70)
    print(" GERAÇÃO DE CURVAS ROC E AUC")
    print("="*70)

    # Solicitar nome do arquivo CSV com dados de ROC
    import glob
    arquivos_roc = glob.glob('roc_data_*.csv')

    if not arquivos_roc:
        print("\nERRO: Nenhum arquivo roc_data_*.csv encontrado!")
        print("Execute primeiro o experimento.py para gerar os dados.")
        return

    # Usar o arquivo mais recente
    arquivo_roc_mais_recente = max(arquivos_roc)
    print(f"\nUsando arquivo ROC: {arquivo_roc_mais_recente}")

    # Carregar dados ROC
    roc_data = carregar_dados_roc(arquivo_roc_mais_recente)

    # Carregar métricas (F1 Score e Acurácia)
    arquivos_resultados = glob.glob('resultados_ml_*.csv')
    if not arquivos_resultados:
        print("\nERRO: Nenhum arquivo resultados_ml_*.csv encontrado!")
        return

    arquivo_resultados_mais_recente = max(arquivos_resultados)
    print(f"Usando arquivo de métricas: {arquivo_resultados_mais_recente}")

    metricas = carregar_metricas_csv(arquivo_resultados_mais_recente)

    # Gerar curva ROC verdadeira
    print("\n" + "="*70)
    print(" CURVA ROC VERDADEIRA (baseada em probabilidades)")
    print("="*70)
    gerar_curvas_roc(roc_data)

    # Gerar gráficos de barras para F1 Score e Acurácia
    print("\n" + "="*70)
    print(" GRÁFICOS DE COMPARAÇÃO DE MÉTRICAS")
    print("="*70)
    gerar_graficos_barras(metricas)

    print("\n" + "="*70)
    print(" VISUALIZAÇÕES GERADAS COM SUCESSO!")
    print("="*70)


if __name__ == "__main__":
    main()
