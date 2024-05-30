from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def cosine_similarity_matrix(embeddings):
    """
    计算句子嵌入的余弦相似度矩阵。
    :param embeddings: 句子嵌入的列表。
    :return: 余弦相似度矩阵。
    """
    return cosine_similarity(embeddings)

def apply_text_rank(similarity_matrix):
    """
    应用TextRank算法计算句子的重要性。
    :param similarity_matrix: 句子之间的相似度矩阵。
    :return: 句子的TextRank得分。
    """
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    return scores

def summarize_clusters(clusters, sentences, embeddings, cluster_weights, top_n=15):
    """
    对每个聚类应用TextRank算法并提取摘要。
    :param clusters: 聚类标签列表。
    :param sentences: 句子列表。
    :param embeddings: 句子嵌入列表。
    :cluster_weight: 权重。
    :param top_n: 每个聚类要提取的摘要句子数。
    :return: 摘要句子列表。
    """
    summary = []
    for cluster_id in set(clusters):
        cluster_sentences = [sentences[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
        cluster_embeddings = [embeddings[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
        cluster_weight = cluster_weights[cluster_id]
        
        if cluster_sentences:
            similarity_matrix = cosine_similarity_matrix(cluster_embeddings)
            scores = apply_text_rank(similarity_matrix)
            ranked_sentences = sorted(((scores[i] * cluster_weight, s) for i, s in enumerate(cluster_sentences)), key=lambda x: x[0], reverse=True)
            top_sentences = [s for _, s in ranked_sentences[:top_n]]
            summary.extend(top_sentences)
    
    return summary
