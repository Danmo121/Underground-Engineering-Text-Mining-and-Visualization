import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

def visualize_clusters_2d(embeddings, clusters):
    tsne_2d = TSNE(n_components=2, random_state=42)
    tsne_result_2d = tsne_2d.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    for i in range(len(set(clusters))):
        cluster_points = tsne_result_2d[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')

    plt.title('词向量聚类可视化 (t-SNE, 2D)')
    plt.xlabel('t-SNE 维度 1')
    plt.ylabel('t-SNE 维度 2')
    plt.legend()
    plt.savefig('t-SNE2D.eps', format='eps')
    plt.show()
    

def visualize_clusters_3d(embeddings, clusters):
    tsne_3d = TSNE(n_components=3, random_state=42)
    tsne_result_3d = tsne_3d.fit_transform(embeddings)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(set(clusters))):
        cluster_points = tsne_result_3d[clusters == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i+1}')

    ax.set_title('词向量聚类可视化 (t-SNE, 3D)')
    ax.set_xlabel('t-SNE 维度 1')
    ax.set_ylabel('t-SNE 维度 2')
    ax.set_zlabel('t-SNE 维度 3')
    ax.legend()
    plt.savefig('t-SNE3D.eps', format='eps')
    plt.show()
