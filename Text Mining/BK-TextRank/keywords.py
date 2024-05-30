from collections import Counter
import itertools
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_keywords(clusters, sentences, embeddings, stop_words, cluster_weights, top_n=35):
    """
    使用TextRank算法直接从每个聚类中提取关键词及其权重，并过滤停用词。
    :param clusters: 聚类标签列表。
    :cluster_weight: 权重。
    :param sentences: 句子列表。
    :param embeddings: 句子嵌入列表。
    :param stop_words: 停用词集合。
    :param top_n: 每个聚类要提取的关键词数量。
    :return: 每个聚类的关键词及其权重。
    """
    cluster_keywords = {}
    for cluster_id in set(clusters):
        cluster_sentences = [sentences[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
        cluster_embeddings = [embeddings[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
        cluster_weight = cluster_weights[cluster_id]

        if cluster_sentences:
            # 构建相似度矩阵
            sim_matrix = cosine_similarity(cluster_embeddings)
            nx_graph = nx.from_numpy_array(sim_matrix)
            scores = nx.pagerank(nx_graph)
            # 将句子与其分数配对，并按分数排序
            ranked_sentences = sorted(((scores[i] * cluster_weight, s) for i, s in enumerate(cluster_sentences)), key=lambda x: x[0], reverse=True)
            # 提取每个句子的关键词，并过滤停用词和单字
            all_words = list(itertools.chain(*[sentence.split() for _, sentence in ranked_sentences]))
            filtered_words = [word for word in all_words if word not in stop_words and len(word) > 1]
            word_scores = Counter(filtered_words)
            cluster_keywords[cluster_id] = [(word, word_scores[word]) for word, _ in word_scores.most_common(top_n)]

    return cluster_keywords
