from transformers import BertTokenizer, BertModel
import torch
import random
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.font_manager import FontProperties


# 确认字体文件路径
# font_path = 'font/simhei.ttf'  # 替换为您的字体路径

# # 加载字体
# font_prop = FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题



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
def get_bert_embedding(word):
    inputs = tokenizer.encode_plus(word, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # 提取 last_hidden_state
    last_hidden_state = outputs[0]

    return last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# 读取语料
with open('input.txt', 'r', encoding='UTF-8') as file:
    corpus = [line.strip() for line in file]

# all_words = [word for sentence in corpus for word in sentence.split() if word not in stopwords]

with open('keyword_BJGSEQ.txt', 'r', encoding='UTF-8') as file:
    predefined_keywords = set(line.strip() for line in file)
predefined_keywords_list = sorted(list(predefined_keywords))


# 创建网络图
G = nx.Graph()
for keyword in predefined_keywords_list:
    keyword_embedding = get_bert_embedding(keyword)
    G.add_node(keyword, embedding=keyword_embedding)

# 计算关键词之间的相似度并添加到图中
similarity_threshold = 0.65 # 设置一个相似度阈值
for i, keyword1 in enumerate(predefined_keywords_list):
    for keyword2 in predefined_keywords_list[i+1:]:
        embedding1 = G.nodes[keyword1]['embedding']
        embedding2 = G.nodes[keyword2]['embedding']
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        if similarity > similarity_threshold:
            G.add_edge(keyword1, keyword2, weight=similarity)


# 计算中心性
degree_centrality = nx.degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# 找到度中心性最高的节点
center_node = max(degree_centrality, key=degree_centrality.get)
# 确保图是连通的，如果不是，可能需要分别处理每个连通分量
if not nx.is_connected(G):
    print("Graph is not connected. Consider applying layout to each connected component separately.")
# 初始化布局字典
pos = {center_node: (0, 0)}
# 使用 BFS 确定节点层级
layers = {}
for node in G:
    try:
        # 使用 try-except 捕获无路径异常
        layers[node] = nx.shortest_path_length(G, source=center_node, target=node)
    except nx.NetworkXNoPath:
        print(f"No path between {center_node} and {node}.")
        # 可以选择忽略这些节点，或给它们指定一个默认的层级
        layers[node] = float('inf')  # 代表无法到达

# 过滤出无穷大之外的层级
filtered_layers = {node: layer for node, layer in layers.items() if layer != float('inf')}

# 计算最大层级
if filtered_layers:
    max_layer = max(filtered_layers.values())
else:
    max_layer = 0

# 对每个层级的节点应用圆形布局
for layer in range(1, max_layer + 1):
    layer_nodes = [node for node, distance in filtered_layers.items() if distance == layer]
    # 调整半径，使得不同层级之间有足够的间距
    radius = layer * 10  # 根据需要调整半径
    layer_pos = nx.circular_layout(layer_nodes, scale=radius, center=(0, 0))
    pos.update(layer_pos)


# 绘制网络图
fig, ax = plt.subplots(figsize=(50, 35))  # 创建一个包含单个轴的图形
pos = nx.spring_layout(G, k=0.5, iterations=10, seed=random_seed)  # k 值调整节点间的距离
# 只绘制权重最高的边
edges = sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1), reverse=True)
edges_to_draw = edges[:int(len(edges) * 1)]  # 只取权重最高的 10%
# 用指定宽度绘制边
edge_width = [0.01 * data['weight'] for u, v, data in edges_to_draw]
nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color='#DCDCDC',width=edge_width)

# higher_similarity_threshold = 0.7 # 设置一个相似度阈值
# high_similarity_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > higher_similarity_threshold]

nx.draw_networkx_nodes(G, pos, node_size=[v * 10000 for v in degree_centrality.values()],
                       node_color=[v for v in closeness_centrality.values()],
                       cmap=plt.cm.viridis)

# nx.draw_networkx_edges(G, pos, edgelist=high_similarity_edges,edge_color= 'gray' ,alpha=0.5, width=1)
# 根据边的权重调整边的粗细
# 假设 edges_to_draw 包含 (node1, node2, data) 元组

nx.draw_networkx_labels(G, pos)
# 边权重标签
# nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=nx.get_edge_attributes(G, 'weight'))
# 添加图例
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(closeness_centrality.values()), vmax=max(closeness_centrality.values())))
sm._A = []
plt.colorbar(sm, ax=ax, label='Closeness Centrality')  # 明确指定 ax

plt.title("Semantic Similarity Network with Centrality Measures")
plt.savefig('BJGSEQ_graph.pdf', format='pdf')



# 将中心性值输出到文件
with open('centrality_output-BJGSEQ.txt', 'w') as out_file:
    out_file.write("Keyword, Degree Centrality, Closeness Centrality, Betweenness Centrality\n")
    for word in predefined_keywords:
        out_file.write(f"{word}, {degree_centrality.get(word, 0)}, {closeness_centrality.get(word, 0)}, {betweenness_centrality.get(word, 0)}\n")

print("Centrality values have been written to centrality_output.txt")
