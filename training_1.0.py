# 确保您已经安装了必要的库
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # 根据您的CUDA版本调整
# pip install transformers sentence-transformers scikit-learn pandas numpy matplotlib seaborn datasets

import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split # 导入 train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import torch
from datasets import load_dataset # 从 Hugging Face datasets 库导入

print("所需库已导入。")

# --- 检查并设置设备 ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU 可用，将使用设备: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU 不可用，将使用 CPU。")

# --- 1. 数据加载与预处理 ---

# 加载 Quora Question Pairs 数据集
# AlekseyKorshuk/quora-question-pairs 只包含 'train' split
qqp_dataset_name = "AlekseyKorshuk/quora-question-pairs"
print(f"正在从 Hugging Face 加载 QQP 数据集：'{qqp_dataset_name}'...")
try:
    # 只加载 'train' split
    qqp_hf_dataset = load_dataset(qqp_dataset_name, split='train')
    qqp_df_full = qqp_hf_dataset.to_pandas() # 将整个 train split 转换为 DataFrame
    print("QQP 数据集加载并转换为 Pandas DataFrame 完成。")
except Exception as e:
    print(f"从 Hugging Face 加载 QQP 数据集 '{qqp_dataset_name}' 时发生错误：{e}")
    print("请检查数据集名称或您的网络连接。")
    print("\n使用模拟 QQP 数据集以继续演示（请替换为您的真实数据）...")
    data = {
        'id': [0, 1, 2, 3, 4],
        'qid1': [1, 2, 3, 4, 5],
        'qid2': [6, 7, 8, 9, 10],
        'question1': [
            "How can I be a good geologist?",
            "What is the best way to learn machine learning?",
            "What is the best way to learn machine learning?",
            "How to lose weight fast?",
            "What are the benefits of eating healthy food?"
        ],
        'question2': [
            "How can I be a good geographer?",
            "What is the best machine learning course?",
            "Is machine learning difficult to learn?",
            "How to gain weight quickly?",
            "Why is healthy food important?"
        ],
        'is_duplicate': [0, 1, 0, 0, 1]
    }
    qqp_df_full = pd.DataFrame(data)

# 清理 QQP 数据：删除包含NaN值的行
qqp_df_full.dropna(inplace=True)
print(f"原始 QQP 数据大小：{len(qqp_df_full)}")

# 将整个数据集转换为 SBERT 需要的 InputExample 格式
full_examples = []
for index, row in qqp_df_full.iterrows():
    full_examples.append(InputExample(texts=[str(row['question1']), str(row['question2'])], label=float(row['is_duplicate'])))

print(f"转换为 InputExample 格式：共 {len(full_examples)} 个样本。")

# 重新手动划分训练集和测试集
# 建议使用 80% 训练，20% 测试（或验证）
train_examples, test_examples = train_test_split(full_examples, test_size=0.2, random_state=42)

print(f"手动划分后：训练集 {len(train_examples)} 个样本，测试集 {len(test_examples)} 个样本。")

# 创建 DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
print(f"训练集 DataLoader 大小：{len(train_dataloader)}")


# --- 加载 StackExchange 数据集作为知识库 (示例) ---
stack_exchange_qa_dataset_name = "PrimeIntellect/stackexchange-question-answering"
print(f"\n正在从 Hugging Face 加载 StackExchange QA 数据集：'{stack_exchange_qa_dataset_name}'...")
try:
    stack_exchange_qa_hf_dataset = load_dataset(stack_exchange_qa_dataset_name, split='train') # 假设它也有train split
    stack_exchange_qa_df = stack_exchange_qa_hf_dataset.to_pandas()
    print("StackExchange QA 数据集加载并转换为 Pandas DataFrame 完成。")
    print(f"StackExchange QA 数据集包含 {len(stack_exchange_qa_df)} 个问答对。")
    print("StackExchange QA 数据集列名:", stack_exchange_qa_df.columns.tolist())
    print("StackExchange QA 数据集前2行示例:")
    print(stack_exchange_qa_df[['prompt', 'gold_standard_solution']].head(2))
except Exception as e:
    print(f"从 Hugging Face 加载 StackExchange QA 数据集 '{stack_exchange_qa_dataset_name}' 时发生错误：{e}")
    print("请检查数据集名称或您的网络连接。")
    stack_exchange_qa_df = pd.DataFrame({'prompt': ["How to connect to a database?", "What is multithreading in Python?"],
                                         'gold_standard_solution': ["Use a database connector library like SQLAlchemy.", "Multithreading allows concurrent execution within a single process."]})
    print("\n使用模拟 StackExchange QA 数据集以继续演示...")


# --- 2. 模型选择与加载 ---
model_name = 'paraphrase-MiniLM-L6-v2'
print(f"\n正在加载预训练模型：{model_name}...")
model = SentenceTransformer(model_name)
print("模型加载完成。")


# --- 3. 模型训练 (微调) ---
train_loss = losses.CosineSimilarityLoss(model=model)

# 评估器使用手动划分的测试集
evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(test_examples, name='qqp-eval')

num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

print(f"\n开始训练模型，Epochs: {num_epochs}, Warmup steps: {warmup_steps}")
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path='sbert_qqp_model',
          evaluation_steps=1000,
          show_progress_bar=True
         )

print("\n模型训练完成。模型已保存到 'sbert_qqp_model' 目录。")


# --- 4. 模型评估 ---
print("\n开始最终评估...")
test_q1_embeddings = model.encode([example.texts[0] for example in test_examples], convert_to_tensor=True)
test_q2_embeddings = model.encode([example.texts[1] for example in test_examples], convert_to_tensor=True)

from torch.nn.functional import cosine_similarity
similarities = cosine_similarity(test_q1_embeddings, test_q2_embeddings).cpu().numpy()
true_labels = np.array([example.label for example in test_examples])

threshold = 0.5
predictions = (similarities >= threshold).astype(int)

accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print(f"评估结果 (使用阈值 {threshold}):")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"F1 分数 (F1 Score): {f1:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(similarities[true_labels == 0], color='red', label='Not Duplicate (0)', kde=True, stat='density', alpha=0.5)
sns.histplot(similarities[true_labels == 1], color='blue', label='Duplicate (1)', kde=True, stat='density', alpha=0.5)
plt.title('Similarity Distribution for Duplicate vs. Non-Duplicate Pairs')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.legend()
plt.show()
print("相似度分布图已生成。")


# --- 5. 模型保存与应用 ---
print("\n演示如何加载保存的模型并进行语义匹配：")
loaded_model = SentenceTransformer('sbert_qqp_model')

historical_questions_from_se = stack_exchange_qa_df['prompt'].astype(str).tolist()

if len(historical_questions_from_se) > 5000:
    historical_questions_from_se = historical_questions_from_se[:5000]
print(f"从 StackExchange QA 数据集中加载了 {len(historical_questions_from_se)} 个问题作为历史知识库。")

print("编码历史问题...")
historical_question_embeddings = loaded_model.encode(historical_questions_from_se, convert_to_tensor=True, show_progress_bar=True)
print("历史问题编码完成。")

user_question = "How to make a connection to my SQL database in Python?"

print(f"\n用户问题: '{user_question}'")
print("查找最相似的历史问题...")

user_question_embedding = loaded_model.encode(user_question, convert_to_tensor=True)

similarities = cosine_similarity(user_question_embedding, historical_question_embeddings).cpu().numpy()

most_similar_idx = np.argmax(similarities)
max_similarity_score = similarities[most_similar_idx]

print(f"最相似的历史问题: '{historical_questions_from_se[most_similar_idx]}'")
print(f"相似度得分: {max_similarity_score:.4f}")

if 'gold_standard_solution' in stack_exchange_qa_df.columns:
    # 找到原始DataFrame中对应问题的索引，这在切割或shuffle后可能需要更复杂的映射
    # 最简单的方法是如果 historical_questions_from_se 保持了原始顺序且没有过滤
    original_idx = stack_exchange_qa_df[stack_exchange_qa_df['prompt'] == historical_questions_from_se[most_similar_idx]].index[0]
    print(f"对应的黄金标准答案: '{stack_exchange_qa_df.loc[original_idx, 'gold_standard_solution']}'")

similarity_threshold = 0.7
if max_similarity_score < similarity_threshold:
    print(f"\n（注意：最高相似度得分 {max_similarity_score:.4f} 低于阈值 {similarity_threshold}，可能需要人工处理或更广泛的搜索。）")
else:
    print(f"\n（相似度得分 {max_similarity_score:.4f} 达到阈值 {similarity_threshold}，被认为是很好的匹配。）")