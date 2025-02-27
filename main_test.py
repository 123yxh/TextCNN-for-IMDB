import re
from torch.optim import Adam
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import gensim.downloader as api
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

# 定义词汇表
class Vocabulary:
    def __init__(self, texts: list[str], min_freq: int = 10):

        # 去除与文字无关的标记，比如说表情符号，链接等等
        text = ' '.join(texts)
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", text)
        text = re.sub("[^a-zA-Z]", " ", text)

        # 删除多余空格
        while '  ' in text:
            text = text.replace('  ', ' ')

        words = text.strip().lower().split()

        c = Counter(words)

        # 创建词汇表
        self.vocabulary = list(set([word for word in words if c[word] >= min_freq])) # 统计词频，只保留频率>=min_freq的词
        self.vocabulary.append('<unk>')  # 添加未知词标记
        #创建索引映射
        self._idx2word = {i: word for i, word in enumerate(self.vocabulary)}
        self._word2idx = {word: i for i, word in enumerate(self.vocabulary)}

    # 获取完整词汇表
    def get_vocabulary(self):
        return self.vocabulary

    # 索引转换为词
    def idx2word(self, idx: int):
        if idx not in self._idx2word:
            return '<unk>'

        return self._idx2word[idx]

    # 词转化为索引
    def word2idx(self, word: str):
        word = word.lower()
        # 对于未知词，统一采用‘<unk>’标签替换
        if word not in self._word2idx:
            return self._word2idx['<unk>']

        return self._word2idx[word]

    # 文本编码--将文本转换为索引序列
    def encode(self, text):
        result = []

        # 每个词遍历
        for word in text.split():
            result.append(self.word2idx(word))

        return result

    # 采用fasttext预训练模型为每个单词转化为词向量
    def build_vectors(self, fasttext):
        vectors = []

        for word in self.vocabulary:
            if fasttext.has_index_for(word):
                vectors.append(fasttext[word])
            else:
                vectors.append(np.zeros(25))

        return np.stack(vectors)

# 将多个样本组合成一个批次-bath
def collate_fn(batch):
    # 将批次中的所有文本序列填充到相同长度
    texts = pad_sequence([b[0] for b in batch], padding_value=pad_idx, batch_first=True)
    # # 将所有标签堆叠在一起
    labels = torch.stack([b[1] for b in batch])

    return texts, labels

# 将原始数据转化为适合训练的text以及label
class IMDB(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab
        self.label2idx = {'0': 0, '1': 1} # 设定映射关系，将消极评论的0转化为数字标签0，积极评论的1转化为数字标签1

    def __getitem__(self, idx):

        # 读取sentences列作为文本输入，labels列作为标签
        text = self.df['sentences'].iloc[idx]
        raw_label = self.df['labels'].iloc[idx]

        if isinstance(raw_label, str):
            label = self.label2idx[raw_label]
        else:
            label = int(raw_label)

        # 转化为torch变量输出
        text = torch.LongTensor(self.vocab.encode(text))
        label = torch.FloatTensor([label])

        return text, label

    def __len__(self):
        return len(self.df)

# 设定TextCNN模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(fs, embedding_dim)),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            for fs in [2, 3, 4, 5]
        ])
        self.fc = nn.Linear(4 * 64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in x]
        x = self.dropout(torch.cat(x, dim=1))
        return self.fc(x)

# 带Drop Out的TextCNN
class TextCNN_DropOut(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx, dropout_rate=0.5):
        super().__init__()
        # 在embedding层后添加dropout
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(fs, embedding_dim)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # 在每个卷积层后添加dropout
                nn.Dropout2d(dropout_rate)
            )
            for fs in [2, 3, 4, 5]
        ])

        self.fc = nn.Linear(4 * 64, 1)
        # 最后的dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 在embedding后应用dropout
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = x.unsqueeze(1)

        # 应用卷积层（包含了dropout）
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in x]

        # 在全连接层之前应用dropout
        x = self.dropout(torch.cat(x, dim=1))
        return self.fc(x)

# 对预测标签与正确标签统计正确率
def binary_accuracy(predicts, y):
    rounded_predicts = torch.round(torch.sigmoid(predicts))
    correct = (rounded_predicts == y).float()
    acc = correct.sum() / len(correct)
    return acc

# 模型训练
def train(model) -> tuple[float, float]:
    model.train()

    train_loss = 0
    train_accuracy = 0
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for x, y in tqdm(train_loader, desc='Train'):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)

        train_loss += loss.item()
        train_accuracy += binary_accuracy(output, y)

        loss.backward()
        optimizer.step()

    # 返回训练的loss以及accuracy
    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    return train_loss, train_accuracy

def train_l1(model) -> tuple[float, float]:
    model.train()

    train_loss = 0
    train_accuracy = 0
    loss_fn = nn.BCEWithLogitsLoss()

    # 调整weight_decay参数来控制L2正则化强度
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for x, y in tqdm(train_loader, desc='Train'):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)

        # 计算主要损失
        loss = loss_fn(output, y)

        # 添加L1正则化
        l1_lambda = 1e-5  # L1正则化系数
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm

        train_loss += loss.item()
        train_accuracy += binary_accuracy(output, y)

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    return train_loss, train_accuracy

# 修改评估函数以返回概率值
@torch.inference_mode()
def evaluate(model, loader) -> tuple[float, float, list, list, list]:
    model.eval()

    total_loss = 0
    total_accuracy = 0
    loss_fn = nn.BCEWithLogitsLoss()

    all_probs = []  # 存储预测概率
    all_preds = []  # 存储二值化的预测
    all_labels = []  # 存储真实标签

    for x, y in tqdm(loader, desc='Evaluation'):
        x, y = x.to(device), y.to(device)

        output = model(x)
        loss = loss_fn(output, y)

        total_loss += loss.item()
        total_accuracy += binary_accuracy(output, y)

        # 获取预测概率
        probs = torch.sigmoid(output).cpu().numpy()
        preds = (probs > 0.5).astype(np.int32)
        labels = y.cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)

    total_loss /= len(loader)
    total_accuracy /= len(loader)

    return total_loss, total_accuracy, all_probs, all_preds, all_labels


# 绘制train_accuracy以及valid_accuracy
def plot_stats(
        train_loss: list[float],
        valid_loss: list[float],
        train_accuracy: list[float],
        valid_accuracy: list[float],
        roc_data: tuple,
        title: str
):
    # 将张量转换为CPU和列表
    train_accuracy = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in train_accuracy]
    valid_accuracy = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in valid_accuracy]

    # 创建包含三个子图的图表
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # 绘制loss图
    ax1.set_title(title + ' Loss')
    ax1.plot(train_loss, label='Train loss')
    ax1.plot(valid_loss, label='Valid loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制accuracy图
    ax2.set_title(title + ' Accuracy')
    ax2.plot(train_accuracy, label='Train accuracy')
    ax2.plot(valid_accuracy, label='Valid accuracy')
    ax2.legend()
    ax2.grid(True)

    # 绘制ROC曲线
    if roc_data is not None:
        fpr, tpr, auc_score = roc_data
        ax3.set_title(f'ROC Curve (AUC = {auc_score:.4f})')
        ax3.plot(fpr, tpr, label=f'ROC curve')
        ax3.plot([0, 1], [0, 1], 'k--')  # 对角线
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.legend()
        ax3.grid(True)

    plt.tight_layout()
    plt.show()

# 完整的训练流程
def whole_train_valid_cycle(model, num_epochs, title):
    valid_probs, valid_labels, valid_preds = [], [], []
    train_loss_history, valid_loss_history = [], []
    train_accuracy_history, valid_accuracy_history = [], []

    for epoch in range(num_epochs):
        # 正则化之后的训练方式
        # train_loss, train_accuracy = train_l1(model)

        train_loss, train_accuracy = train(model)
        valid_loss, valid_accuracy, valid_probs, valid_preds, valid_labels = evaluate(model, test_loader)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        train_accuracy_history.append(train_accuracy)
        valid_accuracy_history.append(valid_accuracy)

        # 打印每个epoch的结果
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')
        print('-' * 50)

    # 计算ROC和AUC
    valid_probs = np.array(valid_probs)
    valid_labels = np.array(valid_labels)
    valid_preds = np.array(valid_preds)

    fpr, tpr, _ = roc_curve(valid_labels, valid_probs)
    auc_score = auc(fpr, tpr)

    # 训练结束后绘制所有图表
    plot_stats(
        train_loss_history, valid_loss_history,
        train_accuracy_history, valid_accuracy_history,
        (fpr, tpr, auc_score),
        title
    )

    # 计算并打印最终的评估指标
    precision = precision_score(valid_labels, valid_preds, average='binary')
    recall = recall_score(valid_labels, valid_preds, average='binary')
    f1 = f1_score(valid_labels, valid_preds, average='binary')

    print('\nFinal Evaluation Metrics:')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC Score: {auc_score:.4f}')

# 加载向量库
fasttext = api.load('glove-twitter-25')

# 读取数据集并使用 Pandas 的方法分割 DataFrame
df = pd.read_csv('datasets.csv')
df = df.iloc[1:]    #跳过第一行标题
df_train = df.iloc[:45000]  # 前 45000 行作为训练集
df_test = df.iloc[45000:]   # 剩下的行作为测试集
df_train = df_train.reset_index(drop=True) # 重置索引
df_test = df_test.reset_index(drop=True)

### 对于训练数据创建词汇表以及向量
train_vocab = Vocabulary(df_train['sentences'].values, min_freq=5)
pad_idx = len(train_vocab.vocabulary)
vectors = train_vocab.build_vectors(fasttext)

# 构建train以及test数据集并加载
train_dataset = IMDB(df_train, train_vocab)
test_dataset = IMDB(df_test, train_vocab)
train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn, pin_memory=True)

# 显示设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# print(torch.cuda.get_device_name())

# 模型定义-TextCNN
model = TextCNN(vocab_size=len(train_vocab.vocabulary) + 1, embedding_dim=25, pad_idx=pad_idx)
# 模型定义-TextCNN带DropOut
# model = TextCNN_DropOut(vocab_size=len(train_vocab.vocabulary) + 1, embedding_dim=25, pad_idx=pad_idx)
model.embedding.weight.data[:len(vectors)] = torch.from_numpy(vectors)
model = model.to(device)

# 开始训练
whole_train_valid_cycle(model, 7, 'Text CNN IMDB')


# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': train_vocab,
    'pad_idx': pad_idx,
    'embedding_dim': 25
}, 'textcnn_model_0226.pth')