import re
from model import *
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 定义词汇表类
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
        self.vocabulary = list(set([word for word in words if c[word] >= min_freq]))
        self.vocabulary.append('<unk>')
        self._idx2word = {i: word for i, word in enumerate(self.vocabulary)}
        self._word2idx = {word: i for i, word in enumerate(self.vocabulary)}

    def get_vocabulary(self):
        return self.vocabulary

    def idx2word(self, idx: int):
        if idx not in self._idx2word:
            return '<unk>'
        return self._idx2word[idx]

    def word2idx(self, word: str):
        word = word.lower()
        if word not in self._word2idx:
            return self._word2idx['<unk>']
        return self._word2idx[word]

    def encode(self, text):
        result = []
        for word in text.split():
            result.append(self.word2idx(word))
        return result

    def build_vectors(self, fasttext):
        vectors = []
        for word in self.vocabulary:
            if fasttext.has_index_for(word):
                vectors.append(fasttext[word])
            else:
                vectors.append(np.zeros(25))
        return np.stack(vectors)

def preprocess_text(text):
    # 清理文本
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", text)
    text = re.sub("[^a-zA-Z]", " ", text)

    # 删除多余空格
    while '  ' in text:
        text = text.replace('  ', ' ')

    return text.strip().lower()


def predict_sentiment(text, model, vocab, device):
    # 预处理文本
    text = preprocess_text(text)

    # 将文本转换为索引序列
    encoded_text = torch.LongTensor([vocab.word2idx(word) for word in text.split()]).unsqueeze(0)
    encoded_text = encoded_text.to(device)

    # 进行预测
    model.eval()
    with torch.no_grad():
        output = model(encoded_text)
        probability = torch.sigmoid(output).item()

    # 返回预测结果和概率
    prediction = 'Positive' if probability >= 0.5 else 'Negative'
    return prediction, probability


def load_model(model_path, device):
    # 加载保存的模型和相关数据
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']
    pad_idx = checkpoint['pad_idx']
    embedding_dim = checkpoint['embedding_dim']

    # 初始化模型
    model = TextCNN(
        vocab_size=len(vocab.vocabulary) + 1,
        embedding_dim=embedding_dim,
        pad_idx=pad_idx
    )

    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model, vocab


def main():
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model, vocab = load_model('textcnn_model_0226.pth', device)

    # 交互式预测
    print("输入 'quit' 退出程序")
    while True:
        text = input("\n请输入要分析的文本: ")
        if text.lower() == 'quit':
            break

        prediction, probability = predict_sentiment(text, model, vocab, device)
        print(f"\n情感分析结果: {prediction}")
        print(f"置信度: {probability:.4f}")


if __name__ == "__main__":
    main()