from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def optimal_num_clusters(embeddings, max_k=10):
    silhouette_scores = []
    for k in range(2, min(len(embeddings), max_k) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        silhouette = silhouette_score(embeddings, labels)
        silhouette_scores.append((k, silhouette))
    optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
    return optimal_k

def cluster_embeddings(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(embeddings)
    return kmeans.labels_

def calculate_similarity_matrix(features, clusters, best_k):
    cluster_centers = []
    for k in range(best_k):
        cluster_features = features[clusters == k]
        cluster_center = np.mean(cluster_features, axis=0)
        cluster_centers.append(cluster_center)
    cluster_centers = np.array(cluster_centers)
    similarity_matrix = cosine_similarity(cluster_centers)
    return similarity_matrix

def calculate_weights(similarity_matrix, threshold=0.5):# threshold可变，0.5是60°的余弦值
    # Calculate weights based on similarity matrix
    weights = []
    for i in range(len(similarity_matrix)):
        similar_clusters = similarity_matrix[i] > threshold
        weight = 1.0 / (1 + np.sum(similar_clusters) - 1)
        weights.append(weight)
    return weights
