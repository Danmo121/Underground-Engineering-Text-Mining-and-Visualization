from text_preprocessing import preprocess, write_text_file
from feature_extraction import get_sentence_embeddings, load_model
from clustering import optimal_num_clusters, cluster_embeddings, calculate_similarity_matrix, calculate_weights
from summarizer import summarize_clusters
from keywords import extract_keywords
from sklearn.preprocessing import StandardScaler
import torch
from visualization_module import visualize_clusters_2d, visualize_clusters_3d

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #加载模型
    tokenizer, model = load_model(device,'bert-base-chinese')

    # 读取文本文件并预处理
    input_file_path = 'input.txt'  # 替换为您的输入文件路径
    sentences = preprocess(input_file_path)

    # 获取句子嵌入
    embeddings = get_sentence_embeddings(sentences, tokenizer, model, device, max_length=512)

    # 确定最佳聚类数
    num_clusters = optimal_num_clusters(embeddings)
    print(num_clusters)
    
    # 应用K-Means聚类
    clusters = cluster_embeddings(embeddings, num_clusters)

    # 计算相似性矩阵和权重
    similarity_matrix = calculate_similarity_matrix(embeddings, clusters, num_clusters)
    cluster_weights = calculate_weights(similarity_matrix)

    # 可视化聚类 (2D)
    visualize_clusters_2d(embeddings, clusters)
    # 可视化聚类 (3D)
    visualize_clusters_3d(embeddings, clusters)

    # 应用TextRank算法并生成摘要
    summary_sentences = summarize_clusters(clusters, sentences, embeddings, cluster_weights)


    def load_stop_words(file_path):
        """
        从文件中加载停用词表。
        :param file_path: 停用词文件的路径。
        :return: 包含停用词的集合。
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            stop_words = set(file.read().splitlines())
        return stop_words
    # 加载停用词表
    stop_words_file_path = 'stopwords.txt'  # 替换为您的停用词文件路径
    stop_words = load_stop_words(stop_words_file_path)

    # 提取关键词
    keywords = extract_keywords(clusters, sentences, embeddings, stop_words, cluster_weights)

    def process_sentence(sentence):
              return ''.join(sentence.split())
    processed_summary_sentences = [process_sentence(sentence) for sentence in summary_sentences]
    summary = "\n".join(processed_summary_sentences)
    print(summary)
    print(keywords)


    def write_text_file(file_path, text):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
    # 调用函数，将摘要写入文件
    output_file_path = 'summary-input.txt'  # 替换为您的输出文件路径
    write_text_file(output_file_path, summary)

    output_keywords_path = 'keywords-input.txt'  # 您的关键词输出文件路径
    with open(output_keywords_path, 'w', encoding='utf-8') as file:
        for cluster_id, cluster_keywords in keywords.items():
            file.write(f"Cluster {cluster_id} Keywords:\n")
            for word, weight in cluster_keywords:
                file.write(f"{word}: {weight}\n")
            file.write("\n")
    print(f"Keywords saved to {output_keywords_path}")



