import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer

# 设置设备为单GPU或CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 读取txt文件中的预处理后的文本
dataset = []
txt_file_path = r"all.txt"
with open(txt_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        dataset.append(line.strip())

# Step 1 - Extract embedding 词嵌入(all-MiniLM-L6-v2或all-roberta-large-v1)
embedding_model = SentenceTransformer(
    "sentence-transformers/all-roberta-large-v1",
    device=device
)

# 基于模型对文本进行语义向量构建
embeddings = embedding_model.encode(dataset, show_progress_bar=True)

# 保存嵌入到文件
embedding_file_path = r"embeddings.npy"
np.save(embedding_file_path, embeddings)
print(f"Embeddings saved to {embedding_file_path}")
