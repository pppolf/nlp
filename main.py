import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import csv

# 导入我们之前写的模块
from utils.data_loader import DataManager
from models import make_model

# =============================================================================
# 0. 辅助函数：设置随机种子 (保证实验可复现)
# =============================================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

# =============================================================================
# 1. 训练与评估函数
# =============================================================================
def train(model, iterator, optimizer, criterion, device, model_type):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(iterator, desc="Training", leave=False)
    
    for batch in pbar:
        # 1. 获取数据并移动到 GPU
        # 注意：data_loader 里的 batch 是一个字典
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # 2. 前向传播 (根据模型类型处理输入)
        if model_type == 'bert':
            mask = batch['attention_mask'].to(device)
            predictions = model(input_ids, mask)
        else:
            # RNN/Transformer 只需要 input_ids
            predictions = model(input_ids)
            
        # 3. 计算 Loss 和 Accuracy
        loss = criterion(predictions, labels)
        
        # 计算准确率 (取最大概率的索引)
        preds = predictions.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # 4. 反向传播与优化
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        # 更新进度条显示的当前 loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device, model_type):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad(): # 评估模式不计算梯度，节省显存
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            if model_type == 'bert':
                mask = batch['attention_mask'].to(device)
                predictions = model(input_ids, mask)
            else:
                predictions = model(input_ids)
                
            loss = criterion(predictions, labels)
            preds = predictions.argmax(dim=1)
            acc = (preds == labels).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# =============================================================================
# 2. 绘图函数 (直接生成论文可用的图)
# =============================================================================
def plot_metrics(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建画布
    plt.figure(figsize=(12, 5))
    
    # 子图1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 子图2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'result_plot.png'))
    print(f"📊 Plot saved to {save_path}/result_plot.png")

# =============================================================================
# 3. 主函数
# =============================================================================
def main():
    # A. 命令行参数定义
    parser = argparse.ArgumentParser(description='NLP Model Experiment')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'ag_news', 'sst2'], help='Dataset name')
    parser.add_argument('--max_len', type=int, default=256, help='Max sequence length')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='lstm', choices=['rnn', 'lstm', 'gru', 'transformer', 'bert'], help='Model architecture')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (use 2e-5 for BERT)')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # B. 初始化设置
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存结果的文件夹
    result_dir = f"results/{args.dataset}_{args.model_type}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    print(f"{'='*30}")
    print(f"🚀 Experiment: {args.model_type.upper()} on {args.dataset.upper()}")
    print(f"💻 Device: {device}")
    print(f"📂 Results will be saved to: {result_dir}")
    print(f"{'='*30}")

    # C. 加载数据
    data_manager = DataManager(args)
    train_loader, test_loader, output_dim, vocab_size = data_manager.load_data()

    # D. 构建模型
    model = make_model(args, vocab_size, output_dim, device)
    
    # 计算参数量
    count_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 The model has {count_parameters:,} trainable parameters")

    # E. 定义优化器和损失函数
    # BERT 通常需要更小的学习率 (如 2e-5)，RNN 可以用 1e-3
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 将模型和损失函数移动到设备
    model = model.to(device)
    criterion = criterion.to(device)

    # F. 训练循环
    # 定义日志文件路径
    log_file = 'epoch_logs.csv'
    
    # 如果是第一次运行，写入表头 (Dataset, Model, Epoch, Accuracy)
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset', 'Model', 'Epoch', 'Accuracy'])

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_valid_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, args.model_type)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion, device, args.model_type)
        
        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入：数据集名, 模型名, 当前Epoch(从1开始), 当前准确率
            writer.writerow([args.dataset, args.model_type, epoch + 1, valid_acc])

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(valid_loss)
        history['val_acc'].append(valid_acc)
        
        epoch_end = time.time()
        epoch_mins, epoch_secs = divmod(epoch_end - epoch_start, 60)
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(result_dir, 'best_model.pt'))
            saved_msg = "🔥 (Saved)"
        else:
            saved_msg = ""
            
        print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% {saved_msg}')
    
    total_time = time.time() - start_time
    print(f"🏁 Training finished in {int(total_time/60)}m {int(total_time%60)}s")

    # G. 绘图与保存
    plot_metrics(history, result_dir)

if __name__ == '__main__':
    main()