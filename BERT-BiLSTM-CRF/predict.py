import torch
import os
from model import BertSeg
from transformers import BertTokenizer
from data_loader import SegDataset
import config
import re

def split_sentence(sentence, max_len, sep_chars=['。', '；', ';', '！', '!', '？', '?']):
    """按最大长度切分句子，优先按标点符号切分"""
    sentence = sentence.replace(" ", "")  # 移除所有空格
    if len(sentence) <= max_len:
        return [sentence]
    else:
        sub_sentences = []
        current_sentence = ""
        for char in sentence:
            current_sentence += char
            if char in sep_chars and len(current_sentence) <= max_len:
                sub_sentences.append(current_sentence)
                current_sentence = ""
            elif len(current_sentence) >= max_len:
                # 尝试找到最后一个分割符
                sep_indices = [current_sentence.rfind(sep) for sep in sep_chars if sep in current_sentence]
                if sep_indices:
                    last_sep_index = max(sep_indices)
                    sub_sentences.append(current_sentence[:last_sep_index + 1])
                    current_sentence = current_sentence[last_sep_index + 1:]
                else:
                    # 没有找到分割符，按最大长度强制切分
                    sub_sentences.append(current_sentence[:max_len])
                    current_sentence = current_sentence[max_len:]
        if current_sentence:
            sub_sentences.append(current_sentence)
        return sub_sentences

def predict(model, sentence, config):
    model.eval()  # 设置模型为评估模式
    # print(f"原始句子长度: {len(sentence)}")  # 打印原始句子长度
    split_sentences = split_sentence(sentence, config.max_len - 5)  # 留出一些空间以防溢出
    # print(f"切分后的句子数量: {len(split_sentences)}")  # 调试信息

    final_predictions = []
    with torch.no_grad():
        for sub_sentence in split_sentences:
            # print("Processing sub-sentence:", sub_sentence)

            dataset = SegDataset([sub_sentence], ["O" * len(sub_sentence)], config)
            input_data = dataset.collate_fn([dataset[0]])
            input_ids, input_token_starts, _, _ = input_data
            input_ids = input_ids.to(config.device)
            input_token_starts = input_token_starts.to(config.device)

            # print("input_ids:", input_ids)

            outputs = model((input_ids, input_token_starts))
            logits = outputs[0]
            predicted_labels = model.crf.decode(logits)
            final_predictions.extend(predicted_labels[0])

    return final_predictions


def labels_to_words(sentence, label_names):
    words = []
    word = ""
    for char, label in zip(sentence, label_names):
        if label == "B" or (label == "M" and not word):  # 新词的起始字符或单个字符的情况
            if word:
                words.append(word)
            word = char
        elif label == "M" or label == "E":  # 连续多个字符
            word += char
        elif label == "S":  # 单独成词
            if word:
                words.append(word)
            words.append(char)
            word = ""
    if word:
        words.append(word)
    return words


if __name__ == "__main__":
    model = BertSeg.from_pretrained(config.model_dir).to(config.device)
    if config.device == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    file_path = "iutput.txt"  # 输入文件路径
    output_file_path = "Seg_output.txt"  # 输出文件路径，可以根据需要修改

    segmented_sentences = []
    error_messages = []  # 存储错误信息

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = line.strip()
            if sentence:
                try:
                    label_ids = predict(model, sentence, config)
                    label_names = [config.id2label[id] for id in label_ids]
                    words = labels_to_words(sentence, label_names)
                    segmented_sentence = " ".join(words)
                    # print(segmented_sentence)
                    segmented_sentences.append(segmented_sentence)
                except Exception as e:
                    error_messages.append(f"处理句子时出错: {sentence}\n错误信息: {e}")

    # 将分词结果写入文件
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        outfile.write("\n".join(segmented_sentences))

    # 打印所有错误信息
    if error_messages:
        print("在处理过程中发生了以下错误：")
        for error in error_messages:
            print(error)

    print(f"分词完成，结果已写入文件：{os.path.abspath(output_file_path)}。")
