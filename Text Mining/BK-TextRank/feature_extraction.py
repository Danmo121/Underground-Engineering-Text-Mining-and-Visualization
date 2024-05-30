import torch
from transformers import BertTokenizer, BertModel
import numpy as np

def load_model(device, local_model_path):
    """
    加载本地预训练的BERT模型和分词器，并将模型移动到指定设备。
    
    参数:
    device (torch.device): 目标设备 (CPU 或 GPU)。
    local_model_path (str): 预训练模型的本地文件夹路径。
    
    返回:
    tokenizer (BertTokenizer): 加载的BERT分词器。
    model (BertModel): 加载的BERT模型。
    """
    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    model = BertModel.from_pretrained(local_model_path).to(device)
    
    return tokenizer, model

def get_sentence_embeddings(sentences, tokenizer, model, device, max_length=512):
    # 将模型设置为评估模式
    model.eval()
    model.to(device)
    
    
    """
    如果没有GPU设备的话
    # 移动输入到目标设备
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 计算句子的平均嵌入表示
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    
    return embeddings
    """
    
    all_embeddings = []
    
    for sentence in sentences:
        inputs = tokenizer(sentence, padding='max_length', truncation=True, return_tensors="pt", max_length=max_length).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 在GPU上计算句子的平均嵌入表示
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
        # 将嵌入从GPU传输回CPU
        embeddings = embeddings.detach().cpu().numpy()
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings
