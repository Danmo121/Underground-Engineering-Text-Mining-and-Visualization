from transformers import BertTokenizer, BertModel
import torch
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 设置随机种子
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
# 加载 BERT 模型和分词器
def load_model(local_model_path):
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    model = BertModel.from_pretrained(local_model_path)
    return tokenizer, model

tokenizer, model = load_model('bert-base-chinese')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer, model = load_model(device,'bert-base-chinese')
# 将词语转化为 BERT 向量
def get_bert_embedding(word):
    inputs = tokenizer.encode_plus(word, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # 提取 last_hidden_state
    last_hidden_state = outputs[0]

    return last_hidden_state.mean(dim=1).squeeze().detach().numpy()
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        stopwords = set([line.strip() for line in file])
    return stopwords

stopwords = load_stopwords('stopwords.txt')  # 替换为您的停用词文件路径



# 读取语料
with open('input.txt', 'r', encoding='UTF-8') as file:
    corpus = [line.strip() for line in file]

all_words = [word for sentence in corpus for word in sentence.split() if word not in stopwords]

from collections import Counter

# 计算词频
word_freq = Counter(all_words)

# 选择高频词，例如选择出现频率最高的前100个词
num_keywords = 25
keywords = sorted([word for word, freq in word_freq.most_common(num_keywords)])
print(keywords)


# 创建网络图
G = nx.Graph()

# 使用BERT词嵌入和高频词
for keyword in keywords:
    keyword_embedding = get_bert_embedding(keyword)
    G.add_node(keyword, embedding=keyword_embedding)

# 计算关键词之间的相似度并添加到图中
similarity_threshold = 0.7  # 设置一个相似度阈值
for i, keyword1 in enumerate(keywords):
    for keyword2 in keywords[i+1:]:
        embedding1 = G.nodes[keyword1]['embedding']
        embedding2 = G.nodes[keyword2]['embedding']
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        if similarity > similarity_threshold:
            G.add_edge(keyword1, keyword2, weight=similarity)


# 计算中心性
degree_centrality = nx.degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# 绘制网络图
# 绘制网络图
fig, ax = plt.subplots(figsize=(25, 15))  # 创建一个包含单个轴的图形
# pos = nx.spring_layout(G)  # 设置布局
pos = nx.spring_layout(G, seed=random_seed)

nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_size=[v * 10000 for v in degree_centrality.values()],
                       node_color=[v for v in closeness_centrality.values()],
                       cmap=plt.cm.viridis)

nx.draw_networkx_edges(G, pos, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax)

# 边权重标签
nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=nx.get_edge_attributes(G, 'weight'))

# 添加图例
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(closeness_centrality.values()), vmax=max(closeness_centrality.values())))
sm._A = []
plt.colorbar(sm, ax=ax, label='Closeness Centrality')  # 明确指定 ax

plt.title("Semantic Similarity Network with Centrality Measures")
plt.savefig('network_graph.eps', format='eps')
plt.show()


# 将中心性值输出到文件
with open('centrality_output.txt', 'w') as out_file:
    out_file.write("Keyword, Degree Centrality, Closeness Centrality, Betweenness Centrality\n")
    for word in keywords:
        out_file.write(f"{word}, {degree_centrality.get(word, 0)}, {closeness_centrality.get(word, 0)}, {betweenness_centrality.get(word, 0)}\n")

print("Centrality values have been written to centrality_output.txt")
