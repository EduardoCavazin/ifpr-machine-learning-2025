import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans


def load_and_preprocess_image(image_path):
    """
    Carrega imagem de arquivo
    """
    import os

    # Normaliza o caminho para o sistema operacional
    image_path = os.path.normpath(image_path)

    print(f"Tentando carregar: {image_path}")
    print(f"Arquivo existe? {os.path.exists(image_path)}")

    # Lê a imagem diretamente com cv2.imread
    im = cv2.imread(image_path)

    if im is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

    # Converte BGR para RGB
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    original_shape = im.shape
    print(f"Imagem carregada com sucesso!")
    print(f"Dimensões da imagem: {original_shape}")

    return im, original_shape


def analyze_hsv_colors(image):
    """
    Transforma imagem para HSV e analisa cores únicas
    """
    print("\n=== ANÁLISE DE CORES (HSV) ===")

    # Converte RGB para HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extrai o canal H (Hue - matiz/cor)
    hue_channel = image_hsv[:, :, 0]

    # Encontra cores únicas
    unique_hues = np.unique(hue_channel)

    print(f"Total de cores únicas (Hue): {len(unique_hues)}")
    print(f"Range de Hue: {unique_hues.min()} - {unique_hues.max()}")

    return image_hsv, unique_hues


def flatten_image(image):
    all_pixels = image.reshape((-1, 3))
    print(f"Total de pixels: {all_pixels.shape[0]}")

    return all_pixels


def find_optimal_k(pixels, k_range=range(2, 11)):
    """
    Encontra o melhor K usando o método do cotovelo (elbow method)
    """
    print("\n=== ENCONTRANDO MELHOR K (Elbow Method) ===")

    inertias = []
    k_values = list(k_range)

    for k in k_values:
        print(f"Testando K={k}...")
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(pixels)
        inertias.append(km.inertia_)

    # Plota o gráfico do erro (inércia)
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (K)', fontsize=12)
    plt.ylabel('Inércia (Erro)', fontsize=12)
    plt.title('Método do Cotovelo - Encontrando o Melhor K', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    # Adiciona valores de inércia nos pontos
    for k, inertia in zip(k_values, inertias):
        plt.annotate(f'{inertia:.0f}',
                     xy=(k, inertia),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     fontsize=9)

    plt.tight_layout()
    plt.show()

    print("\nValores de inércia por K:")
    for k, inertia in zip(k_values, inertias):
        print(f"  K={k}: {inertia:.2f}")

    return k_values, inertias


def apply_kmeans(pixels, n_clusters=4):
    print(f"\nAplicando K-Means com {n_clusters} clusters...")

    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(pixels)

    centers = np.array(km.cluster_centers_, dtype='uint8')

    print("Cores dominantes encontradas (RGB):")
    print(centers)

    return km, centers


def plot_dominant_colors(centers):
    n_colors = len(centers)
    plt.figure(figsize=(8, 2))

    for i, color in enumerate(centers):
        plt.subplot(1, n_colors, i + 1)
        plt.axis("off")

        color_swatch = np.zeros((100, 100, 3), dtype='uint8')
        color_swatch[:, :, :] = color
        plt.imshow(color_swatch)
        plt.title(f"Cor {i+1}")

    plt.suptitle("Cores Dominantes", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()


def segment_image(km, centers, original_shape):
    print("\nSegmentando a imagem...")

    n_pixels = original_shape[0] * original_shape[1]
    new_img = np.zeros((n_pixels, 3), dtype='uint8')

    for ix in range(n_pixels):
        new_img[ix] = centers[km.labels_[ix]]

    new_img = new_img.reshape(original_shape)

    return new_img


def plot_comparison(original_image, segmented_image, k_value):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Imagem Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title(f"Imagem Segmentada (K={k_value})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """
    Função principal seguindo os requisitos da proposta
    """
    import os

    # ====== CONFIGURAÇÕES ======
    # Usa o diretório do script para construir o caminho absoluto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    IMAGE_PATH = os.path.join(script_dir, "TesteML.jpg")

    K_RANGE = range(2, 11)  # Range de K para testar

    print("=" * 60)
    print("TRABALHO: SEGMENTAÇÃO DE IMAGENS COM K-MEANS")
    print("=" * 60)

    # ====== ETAPA 1: CARREGAR IMAGEM ======
    print("\n[ETAPA 1] Carregando imagem...")
    image, original_shape = load_and_preprocess_image(IMAGE_PATH)

    # ====== ETAPA 2: ANÁLISE HSV E CORES ÚNICAS ======
    print("\n[ETAPA 2] Analisando esquema de cores...")
    image_hsv, unique_hues = analyze_hsv_colors(image)

    # Mostra a imagem original com informação de cores únicas
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Imagem Original - {len(unique_hues)} valores H únicos")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ====== ETAPA 3: ENCONTRAR MELHOR K ======
    print("\n[ETAPA 3] Simplificando imagem com K-Means...")

    # Achatar imagem para aplicar K-Means
    pixels = flatten_image(image)

    # Encontra o melhor K usando elbow method
    k_values, inertias = find_optimal_k(pixels, k_range=K_RANGE)

    # Solicita ao usuário o melhor K (baseado no gráfico)
    print("\n" + "=" * 60)
    print("Analisando o gráfico acima, escolha o melhor valor de K")
    print("(procure pelo 'cotovelo' na curva)")
    best_k = int(input("Digite o melhor K: "))

    # ====== ETAPA 4: GERAR IMAGEM FINAL COM MELHOR K ======
    print(f"\n[ETAPA 4] Gerando imagem final com K={best_k}...")

    # Aplica K-Means com o melhor K
    km, centers = apply_kmeans(pixels, n_clusters=best_k)

    # Plota cores dominantes
    plot_dominant_colors(centers)

    # Segmenta imagem
    segmented_image = segment_image(km, centers, original_shape)

    # Plota comparação final
    plot_comparison(image, segmented_image, best_k)

    print("\n" + "=" * 60)
    print("SEGMENTAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"Imagem simplificada usando {best_k} cores")
    print("=" * 60)


if __name__ == "__main__":
    main()
