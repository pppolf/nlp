import os
import torch
import torch.nn as nn
from transformers import BertModel

# =============================================================================
# 1. 统一的 RNN/LSTM/GRU 模型
# =============================================================================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, 
                 n_layers=2, bidirectional=True, dropout=0.5, model_type='lstm'):
        super().__init__()
        self.model_type = model_type.lower()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 定义循环层
        # batch_first=True 让输入变为 [batch, seq_len, dim]
        rnn_args = {
            'input_size': embed_dim,
            'hidden_size': hidden_dim,
            'num_layers': n_layers,
            'bidirectional': bidirectional,
            'batch_first': True,
            'dropout': dropout if n_layers > 1 else 0
        }
        
        if self.model_type == 'lstm':
            self.rnn = nn.LSTM(**rnn_args)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(**rnn_args)
        else: # Basic RNN
            self.rnn = nn.RNN(**rnn_args)
            
        # 全连接层 (分类器)
        # 如果是双向，hidden_dim 需要 x2
        self.fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(self.fc_input_dim, output_dim)
        
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text: [batch_size, seq_len]
        
        # 1. Embedding
        embedded = self.dropout(self.embedding(text)) # [batch, seq_len, embed]
        
        # 2. RNN Layer
        # output: 每个时间步的输出
        # hidden: 最后一个时间步的隐藏状态 (用于分类)
        if self.model_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)
            
        # 3. 获取最后一个隐藏状态
        # hidden shape: [num_layers * num_directions, batch, hidden_dim]
        # 我们需要把双向的最后两个 hidden 拼接起来
        if self.rnn.bidirectional:
            # 取最后两层 (正向最后一层 + 反向最后一层)
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
            
        # 4. Classification
        return self.fc(self.dropout(hidden))

# =============================================================================
# 2. Transformer (Encoder Only) 模型
# =============================================================================
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, hidden_dim, n_layers, output_dim, max_len=256, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 简单的位置编码 (Learnable Positional Encoding)
        # 也可以用 sin/cos 固定编码，但 Learnable 在大作业里更简单且效果够用
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text: [batch, seq_len]
        batch_size, seq_len = text.shape
        
        # 1. Embedding + Positional Encoding
        # 截取对应长度的位置编码
        pos_embed = self.pos_embedding[:, :seq_len, :]
        embedded = self.dropout(self.embedding(text) + pos_embed)
        
        # 2. Transformer Forward
        # mask 用于忽略 padding (0) 的位置，避免 attention 关注到 padding
        # src_key_padding_mask: [batch, seq_len] (True for padding)
        padding_mask = (text == 0)
        
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        # shape: [batch, seq_len, embed_dim]
        
        # 3. Pooling (聚合策略)
        # 这里使用 Mean Pooling (取所有非 padding 词向量的平均值)
        # 为了简单，直接对所有输出取平均 (稍微粗糙但有效)
        output = transformer_out.mean(dim=1) 
        
        return self.fc(output)

# =============================================================================
# 3. BERT 模型
# =============================================================================
class BERTClassifier(nn.Module):
    def __init__(self, output_dim, cache_dir=None, freeze_bert=False):
        super().__init__()
        # 自动下载/加载预训练权重
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
            cache_dir=cache_dir
        )
        
        # 是否冻结 BERT 参数 (只训练最后的分类层)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        # 这里的 hidden_size 通常是 768
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        # BERT forward
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # pooler_output 是 [CLS] token 经过一层线性层后的输出
        # 专门设计用于分类任务
        return self.fc(outputs.pooler_output)

# =============================================================================
# 4. 工厂函数：一键生成模型
# =============================================================================
def make_model(args, vocab_size, output_dim, device):
    """
    根据命令行参数 args 自动选择并初始化模型
    """
    model_type = args.model_type.lower()
    pre_model_path = os.path.join(os.getcwd(), 'pre-model')
    
    print(f"🏗️ Building Model: {model_type.upper()}...")
    
    if model_type == 'bert':
        model = BERTClassifier(output_dim, cache_dir=pre_model_path)
        
    elif model_type in ['rnn', 'lstm', 'gru']:
        # 可以在 args 里加这些参数，这里先给默认值
        model = RNNClassifier(
            vocab_size=vocab_size,
            embed_dim=100,      # 词向量维度
            hidden_dim=256,     # 隐藏层维度
            output_dim=output_dim,
            n_layers=2,         # 层数
            bidirectional=True, # 双向
            dropout=0.5,
            model_type=model_type
        )
        
    elif model_type == 'transformer':
        # 注意: embed_dim 必须能被 n_heads 整除
        # 这里强制设 embed_dim=128, n_heads=4
        model = TransformerClassifier(
            vocab_size=vocab_size,
            embed_dim=128,      
            n_heads=4,
            hidden_dim=256,
            n_layers=2,
            output_dim=output_dim,
            max_len=args.max_len,
            dropout=0.5
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)