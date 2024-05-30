from collections import defaultdict
import math

def loadStopWords(stopWordsFilePath):
    stopwords = set()
    with open(stopWordsFilePath, 'r', encoding='utf8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords
def loadDataSet(stopwords):
    dataset = []
    wordCount = defaultdict(int) # 记录每个词的总频率
    wordDocCount = defaultdict(int) # 记录包含特定词的文档数

    with open('input.txt', 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip('\n\ufeff').split(' ')
            filtered_words = set(word for word in line if word not in stopwords) # 使用集合去重

            for word in filtered_words:
                wordCount[word] += 1 # 更新总词频
                wordDocCount[word] += 1 # 更新文档频率

            dataset.append(filtered_words)

    return dataset, wordCount, wordDocCount

def feature_select(wordCount, wordDocCount, doc_num):
    word_tf_idf = {}
    for word in wordCount:
        tf = wordCount[word] / sum(wordCount.values()) # 计算TF
        idf = math.log(doc_num / (wordDocCount[word] + 1)) # 计算IDF
        word_tf_idf[word] = round(tf * idf, 3) # 计算TF-IDF并四舍五入

    return sorted(word_tf_idf.items(), key=lambda x: x[1], reverse=True)

def save_features(features, output_file_path):
    with open(output_file_path, 'w', encoding='utf8') as fw:
        # 可以在这里写入标题或注释
        fw.write("Word\tTF-IDF\n")

        for word, score in features:
            # 使用格式化字符串保持输出整洁
            fw.write(f"{word}\t{score}\n")

# 主程序部分
stopwords = loadStopWords('stopwords.txt')  # 加载停用词文件
data_list, wordCount, wordDocCount = loadDataSet(stopwords)  # 加载数据并移除停用词
features = feature_select(wordCount, wordDocCount, len(data_list))  # 计算TF-IDF


# 保存特征到文件
output_file_path = 'output.txt'
save_features(features, output_file_path)
