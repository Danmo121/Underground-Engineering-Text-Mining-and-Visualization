#-*- encoding:utf-8 -*-
from textrank4zh import TextRank4Keyword, TextRank4Sentence

with open('Cut_BJGSEQ.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 将文本转换为按空格分隔的列表
words = text.split()

# 自定义的关键词提取过程
tr4w = TextRank4Keyword()
tr4w.analyze(text=' '.join(words), lower=True, window=2, tokenizer=None)

print( '关键词：' )
for item in tr4w.get_keywords(100, word_min_len=1):
    print(item.word, item.weight)

# #保存关键词到文件cond
with open('keyword_BJGSEQ.txt', 'w', encoding='utf-8') as file:
    for item in tr4w.get_keywords(10000, word_min_len=2):
        file.write(f"{item.word} {item.weight}\n")


# 自定义的摘要生成过程
tr4s = TextRank4Sentence()
tr4s.analyze(text=' '.join(words), lower=True, source='all_filters', tokenizer=None)

print()
print( '摘要：' )
with open('summary_BJGSEQ.txt', 'w', encoding='utf-8') as file:
    for item in tr4s.get_key_sentences(num=40):
        summary_sentences = item.sentence.replace(' ', '')
        print(item.index, item.weight,  summary_sentences)
        file.write(f"{item.index} {item.weight} {summary_sentences}\n")

