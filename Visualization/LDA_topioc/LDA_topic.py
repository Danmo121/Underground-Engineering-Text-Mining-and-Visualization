import kwx
from kwx.visuals import t_sne
import matplotlib.pyplot as plt
from kwx.visuals import pyLDAvis_topics
from kwx.visuals import graph_topic_num_evals
import random


input_language = "chinese"  # see kwx.languages for options

def read_stopwords(stopwords_file):
    """从文件中读取停用词列表。"""
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f]
    return set(stopwords)

def remove_stopwords(text, stopwords):
    """从文本中去除停用词。"""
    words = text.split()
    words_filtered = [word for word in words if word not in stopwords]
    return ' '.join(words_filtered)

def prepare_corpus(corpus_file, stopwords_file):
    """读取语料文件并去除停用词。"""
    stopwords = read_stopwords(stopwords_file)
    corpus_cleaned = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_cleaned = remove_stopwords(line.strip(), stopwords)
            corpus_cleaned.append(line_cleaned)
    return corpus_cleaned

def main():
    corpus_file = 'input.txt'  # 语料文件路径
    stopwords_file = 'stopwords.txt'  # 停用词文件路径
    text_corpus = prepare_corpus(corpus_file, stopwords_file)
    # for i, doc in enumerate(text_corpus):
    #     if i < 10:  # 只打印前10个文档
    #         print(doc)
    t_sne(
        dimension="both",  # 2d and 3d are options
        text_corpus=text_corpus,
        num_topics=10,
        remove_3d_outliers=True,
    )
    plt.savefig('input-eva-top_ts.eps', format='eps')
    plt.show()

    graph_topic_num_evals(
        method=["lda"],
        text_corpus=text_corpus,
        num_keywords=100,
        topic_nums_to_compare=list(range(5, 20)),
        metrics=True,  # stability and coherence
    )
    plt.savefig('input-eva-top_num.eps', format='eps')
    plt.show()

    pyLDAvis_topics(
        method="lda",
        text_corpus=text_corpus,
        num_topics=10,
        save_file=True,
        display_ipython=False,  # For Jupyter integration
    )

if __name__ == '__main__':
    main()
