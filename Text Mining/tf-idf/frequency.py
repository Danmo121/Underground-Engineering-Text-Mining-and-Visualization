from collections import Counter
import csv

# 读取停用词文件
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = {line.strip() for line in file}
    return stopwords

# 读取分好词的文本文件并去除停用词
def read_words(file_path, stopwords):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().split()
    filtered_words = [word for word in words if word not in stopwords]
    return filtered_words

# 写入CSV文件
def write_to_csv(word_counts, csv_file_path):
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Word', 'Frequency'])
        for word, count in word_counts.items():
            writer.writerow([word, count])

# 主函数
def main():
    input_file_path = 'input.txt'  # 分好词的文本文件路径
    output_csv_file_path = 'frequencies.csv'  # 输出CSV文件路径
    stopwords_file_path = 'stopwords.txt'  # 停用词文件路径

    stopwords = load_stopwords(stopwords_file_path)
    words = read_words(input_file_path, stopwords)
    word_counts = Counter(words)

    write_to_csv(word_counts, output_csv_file_path)
    print(f"Word frequency written to {output_csv_file_path}")

# 执行主函数
if __name__ == '__main__':
    main()
