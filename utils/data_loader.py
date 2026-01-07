import torch
import os
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer
from collections import Counter

# ==========================================
# 1. 自定义 Dataset 类 (兼容 PyTorch DataLoader)
# ==========================================
class NLPDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # 构造每一个样本的字典
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# ==========================================
# 2. 核心数据管理器类
# ==========================================
class DataManager:
    def __init__(self, args):
        """
        args: 包含命令行参数的对象 (必须包含 .dataset, .model_type, .batch_size, .max_len)
        """
        self.dataset_name = args.dataset
        self.model_type = args.model_type # 'bert' 或 'rnn'/'lstm'/'gru'/'transformer'
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        
        # 存储构建的词表（仅 RNN 模式用到）
        self.vocab = None 
        self.word2idx = None

        self.data_path = os.path.join(os.getcwd(), 'data')

        # [新增] 定义预训练模型保存路径
        self.model_cache_path = os.path.join(os.getcwd(), 'pre-model')
        
        # [新增] 如果不存在则创建
        if not os.path.exists(self.model_cache_path):
            os.makedirs(self.model_cache_path)
            print(f"📁 Created model cache directory: {self.model_cache_path}")

    def load_data(self):
        """
        主函数：下载数据 -> 预处理 -> 返回 Loaders 和关键参数
        Returns:
            train_loader, test_loader, output_dim, vocab_size
        """
        print(f"🔄 Loading dataset: {self.dataset_name}...")
        print(f"📂 Data will be cached at: {self.data_path}")
        
        # -------------------------------------------
        # A. 加载原始数据 (Hugging Face Datasets)
        # -------------------------------------------
        if self.dataset_name == 'imdb':
            raw_dataset = load_dataset("imdb", cache_dir=self.data_path)
            text_col, label_col = 'text', 'label'
            output_dim = 2
        elif self.dataset_name == 'ag_news':
            raw_dataset = load_dataset("ag_news", cache_dir=self.data_path)
            text_col, label_col = 'text', 'label'
            output_dim = 4
        elif self.dataset_name == 'sst2':
            # SST-2 是 GLUE benchmark 的一部分
            raw_dataset = load_dataset("glue", "sst2", cache_dir=self.data_path)
            text_col, label_col = 'sentence', 'label'
            output_dim = 2
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # -------------------------------------------
        # B. 根据模型类型进行分词处理
        # -------------------------------------------
        if self.model_type == 'bert':
            train_dataset, test_dataset = self._process_for_bert(raw_dataset, text_col, label_col)
            vocab_size = 0 # BERT 自带 embedding，不需要我们传入 vocab_size
        else:
            train_dataset, test_dataset, vocab_size = self._process_for_rnn(raw_dataset, text_col, label_col)

        # -------------------------------------------
        # C. 构造 DataLoader
        # -------------------------------------------
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"✅ Data loaded successfully. Output Dim: {output_dim}, Vocab Size: {vocab_size}")
        return train_loader, test_loader, output_dim, vocab_size

    # ================= INTERNAL METHODS =================

    def _process_for_bert(self, dataset, text_col, label_col):
        """BERT 专属处理：使用 Hugging Face Tokenizer"""
        print("⚙️ Processing for BERT (Tokenization & Padding)...")
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', 
            cache_dir=self.model_cache_path
        )

        def tokenize_function(examples):
            return tokenizer(examples[text_col], padding="max_length", truncation=True, max_length=self.max_len)

        # 批量处理
        tokenized_train = dataset['train'].map(tokenize_function, batched=True)
        # 部分数据集验证集名字不同，这里做个简单的兼容
        if self.dataset_name == 'sst2':
            val_split = 'validation'
        else:
            val_split = 'test' if 'test' in dataset else 'validation'
        tokenized_test = dataset[val_split].map(tokenize_function, batched=True)

        # 转换为 PyTorch 格式
        train_ds = NLPDataset(
            encodings={'input_ids': tokenized_train['input_ids'], 'attention_mask': tokenized_train['attention_mask']},
            labels=tokenized_train[label_col]
        )
        test_ds = NLPDataset(
            encodings={'input_ids': tokenized_test['input_ids'], 'attention_mask': tokenized_test['attention_mask']},
            labels=tokenized_test[label_col]
        )
        return train_ds, test_ds

    def _process_for_rnn(self, dataset, text_col, label_col):
        """RNN/LSTM 专属处理：构建词表 + 序列化"""
        print("⚙️ Building Vocabulary for RNN/LSTM...")
        
        # 1. 构建词表 (仅使用训练集)
        tokens_list = [text.lower().split() for text in dataset['train'][text_col]]
        word_counts = Counter([token for line in tokens_list for token in line])
        
        # 只保留出现频率最高的 Top N 词，防止词表爆炸
        MAX_VOCAB_SIZE = 25000 
        most_common = word_counts.most_common(MAX_VOCAB_SIZE)
        
        # 特殊 Token: <PAD> = 0, <UNK> = 1
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            
        vocab_size = len(self.word2idx)

        # 2. 数值化 & Padding 函数
        def encode_and_pad(text_list):
            input_ids = []
            for text in text_list:
                tokens = text.lower().split()
                # 转换: 词 -> ID (不存在则用 UNK)
                ids = [self.word2idx.get(t, 1) for t in tokens]
                
                # 截断或填充
                if len(ids) > self.max_len:
                    ids = ids[:self.max_len]
                else:
                    ids += [0] * (self.max_len - len(ids)) # Padding with 0
                input_ids.append(ids)
            return input_ids

        # 3. 处理训练集和测试集
        train_ids = encode_and_pad(dataset['train'][text_col])
        if self.dataset_name == 'sst2':
            val_split = 'validation'
        else:
            val_split = 'test' if 'test' in dataset else 'validation'
        test_ids = encode_and_pad(dataset[val_split][text_col])

        train_ds = NLPDataset({'input_ids': train_ids}, dataset['train'][label_col])
        test_ds = NLPDataset({'input_ids': test_ids}, dataset[val_split][label_col])
        
        return train_ds, test_ds, vocab_size