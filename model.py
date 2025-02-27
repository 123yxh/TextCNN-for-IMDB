import torch
from torch import nn
from torch.nn import functional as F

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