def preprocess(file_path):
    """
    读取预处理过的文件，每行一个句子，词语以空格分隔。
    :param file_path: 预处理文件的路径。
    :return: 句子的列表。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    # 移除每个句子末尾的换行符并返回
    return [s.strip() for s in sentences]

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_text_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

