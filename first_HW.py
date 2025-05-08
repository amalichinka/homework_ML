import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def calc_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def assign_clusters(data, centroids):
    return np.array([np.argmin([calc_distance(p, c) for c in centroids]) for p in data])


def update_centroids(data, clusters, k):
    return np.array([
        np.mean(data[clusters == i], axis=0) if len(data[clusters == i]) > 0
        else data[np.random.choice(len(data))]
        for i in range(k)
    ])


def plot_clusters(data, clusters, centroids, iteration):
    plt.figure(figsize=(6, 6))
    for i in range(len(np.unique(clusters))):
        plt.scatter(data[clusters == i][:, 0], data[clusters == i][:, 1], label=f'Кластер {i + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Центроиды')
    plt.title(f'Итерация {iteration}')
    plt.legend()
    plt.show()


def final_plt(data, clusters, centroids):
    plt.figure(figsize=(15, 10))
    idx = 1
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            plt.subplot(3, 3, idx)
            idx += 1
            for cluster_id in np.unique(clusters):
                plt.scatter(data[clusters == cluster_id][:, i], data[clusters == cluster_id][:, j],
                            label=f'Кластер {cluster_id + 1}')
            plt.scatter(centroids[:, i], centroids[:, j], s=200, c='black', marker='X')
            plt.xlabel(f'Feature {i + 1}')
            plt.ylabel(f'Feature {j + 1}')
            plt.legend()
    plt.tight_layout()
    plt.show()


def k_means(data, k, max_iter=10):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for i in range(max_iter):
        clusters = assign_clusters(data, centroids)
        plot_clusters(data, clusters, centroids, i + 1)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(centroids, new_centroids):
            print(f"Остановлено на итерации {i + 1}")
            break
        centroids = new_centroids
    final_plt(data, clusters, centroids)


def optimal_k_by_elbow_method(data):
    wcss = []
    silhouette_scores = []

    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), wcss, marker='o', label='WCSS')
    plt.title("Выбор k по методу локтя")
    plt.xlabel("Количество кластеров")
    plt.ylabel("WCSS")
    plt.grid(True)
    plt.show()

    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"Оптимальное количество кластеров : {optimal_k}")

    return optimal_k


def main():
    irises = load_iris()
    data = irises.data
    k = optimal_k_by_elbow_method(data)
    k_means(data, k)


if __name__ == "__main__":
    main()
