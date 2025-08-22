import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import math
import copy
import os
import gc
import time

# Define the Config class, including all configuration parameters
class Config:
    # Data Directory
    data_dir = '.'
    data_path = 'data.npy'
    
    # MMformer configuration
    train_mmformer = False # Whether to train the MMformer model
    load_mmformer = True # Whether to load an existing MMformer model
    save_path = "best_MMformer_model.pth"
    prediction_path = "mmformer_predictions.npy"
    
    # MMformer Visualization
    visualize_predictions = True    # Whether to visualize MMformer prediction results
    visualization_output_dir = 'visualizations'  # Visualization output directory
    
    # MMformer parameters
    mm_d_model = 768
    mm_n_heads = 8
    mm_d_ff = 2048
    mm_e_layers = 6
    
    mm_seq_len = 150
    mm_pred_len = 30
    mm_dropout = 0.2
    mm_factor = 365
    mm_feature_size = 7
    batch_size_mmformer = 128
    num_mc_samples = 1000       # MC次数
    
    mm_max_norn = 1
    meta_batch_size = 128       # 每次元更新基于多少个任务（城市）
    inner_batch_size = 128      # 内循环适应时每个任务使用的批大小
    
    outer_lr_max = 5e-5      # Maximum outer loop learning rate
    outer_lr_min = 5e-8      # Minimum outer loop learning rate
    inner_lr_init = 1e-5      # Initial inner loop learning rate
    inner_lr_min = 1e-8       # Minimum inner loop learning rate
    inner_steps = 10
    lr_inner = 1e-5
    warmup_epochs = 15         # Number of warmup epochs
    lr_scheduler_patience = 10  # Patience value for the learning rate scheduler
    weight_decay = 0.02        # Optimizer Weight Decay
    patience = 20              # Patience
    mm_num_epochs = 700        # number of epochs
    
    #final_div_factor=configs.outer_lr_max/configs.outer_lr_min
    
    epsilon = 1
    #min_actual = 1e-4
    #Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
configs = Config()

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, epsilon=1):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, pred, actual):
        mse = F.mse_loss(pred, actual)
        mape = torch.mean(torch.abs((actual - pred) / (torch.abs(actual) + self.epsilon)))
        return self.alpha * mse + (1 - self.alpha) * mape

# 创建损失函数实例
loss_function = HybridLoss(alpha=0.3, epsilon=1)

# MMformer
# Causal mask
def generate_causal_mask(seq_len, batch_size, n_heads, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    mask = mask.unsqueeze(0).unsqueeze(1)  # 形状: (1, 1, seq_len, seq_len)
    mask = mask.expand(batch_size, n_heads, seq_len, seq_len)  # 形状: (batch_size, n_heads, seq_len, seq_len)
    return mask

# Monte Carlo Dropout
class MCDropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)
    
# Probabilistic Attention
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=7, scale=None, attention_dropout=0.2):
        super(ProbAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = MCDropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, H, L_q, D = queries.shape  # Batch size, Heads, Length, Depth per head
        L_k = keys.shape[2]            # 获取键的序列长度

        # Scale dot-product attention
        scale = self.scale or 1. / (D ** 0.5)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, values)
        return context, attn

class DataEmbeddingInverted(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbeddingInverted, self).__init__()
        # 核心变化：输入维度是序列长度seq_len，而不是特征数
        self.value_embedding = nn.Linear(seq_len, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, x_mark=None):
        # x的原始形状: (B, L, C)
        # B: 批量大小, L: 序列长度, C: 特征数量
        
        # 1. 维度反转
        x = x.permute(0, 2, 1)  # -> (B, C, L)
        
        # 2. 将每个特征的时间序列作为一个Token进行嵌入
        # self.value_embedding的输入是(B, C, L)，输出是(B, C, d_model)
        if x_mark is None:
            x = self.value_embedding(x)  # (B, feature_size, d_model)
        else:
            # 如果有协变量，也需要反转
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
            
        return self.dropout(x)
    
# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        B, L_q, _ = query.size()
        B, L_k, _ = key.size()
        # 线性变换并拆分多头
        query = self.query_projection(query).view(B, L_q, self.n_heads, self.d_k)
        key   = self.key_projection(key).view(B, L_k, self.n_heads, self.d_k)
        value = self.value_projection(value).view(B, L_k, self.n_heads, self.d_v)
    
        # 作 transpose 以便做带多头维度的 matmul
        query = query.permute(0, 2, 1, 3)   # (B, n_heads, L_q, d_k)
        key   = key.permute(0, 2, 1, 3)     # (B, n_heads, L_k, d_k)
        value = value.permute(0, 2, 1, 3)   # (B, n_heads, L_k, d_v)
    
        context, attn = self.attention(query, key, value, attn_mask)
        # context: (B, n_heads, L_q, d_v)
        context = context.permute(0, 2, 1, 3).contiguous()  # (B, L_q, n_heads, d_v)
        context = context.view(B, L_q, -1)                  # (B, L_q, d_model)
        out = self.out_projection(context)                  # (B, L_q, d_model)
        return out, attn

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = MCDropout(dropout)

    def forward(self, src, src_mask=None):
        src2, _ = self.attention_layer(src, src, src, attn_mask=src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

# Encoder
class Encoder(nn.Module):
    def __init__(self, layer, N_max, d_model):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N_max)])
        # self.norm = nn.LayerNorm(layer.size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return self.norm(src)

# MMformer Model
class MMformer(nn.Module):
    def __init__(self, configs):
        super(MMformer, self).__init__()
        self.configs = configs
        self.pred_len = configs.mm_pred_len
        
        # Embedding
        self.enc_embedding = DataEmbeddingInverted(
            configs.mm_seq_len, configs.mm_d_model, configs.mm_dropout
        )
        
        # Encoder
        encoder_layer = EncoderLayer(
            AttentionLayer(
                ProbAttention(False, configs.mm_factor, attention_dropout=configs.mm_dropout),
                configs.mm_d_model,
                configs.mm_n_heads
            ),
            configs.mm_d_model,
            configs.mm_d_ff,
            dropout=configs.mm_dropout
        )
        self.encoder = Encoder(encoder_layer, configs.mm_e_layers, configs.mm_d_model)
    
        self.projector = nn.Linear(configs.mm_d_model, self.pred_len, bias=True)
        self.pred_len = configs.mm_pred_len
        
    def forward(self, src, src_mark=None):
        """
        一次性前向传播 (iTransformer核心逻辑)
        src: (B, L, C)
        """
        # (1) 嵌入与反转: (B, L, C) -> (B, C, d_model)
        enc_in = self.enc_embedding(src, src_mark)
        
        # (2) 编码器处理: (B, C, d_model) -> (B, C, d_model)
        # 注意: 此处注意力在特征维度C上计算，不需要掩码
        enc_out = self.encoder(enc_in, mask=None) 
        
        # (3) 投影到预测长度: (B, C, d_model) -> (B, C, pred_len)
        dec_out = self.projector(enc_out)
        
        # (4) 维度转置回标准格式: (B, C, pred_len) -> (B, pred_len, C)
        return dec_out.permute(0, 2, 1)
    
# === MAML前向传播函数 ===
def forward_with_params(model, src, adapted_params):
    """
    使用 adapted_params (而非原模型参数) 执行一次iTransformer架构的前向传播。
    """
    original_params_data = {name: param.data.clone() for name, param in model.named_parameters()}
    try:
        for name, param in model.named_parameters():
            if name in adapted_params:
                param.data.copy_(adapted_params[name])
        outputs = model.forward(src)
    finally:
        for name, param in model.named_parameters():
            param.data.copy_(original_params_data[name])
    return outputs

def backup_model_params(model):
    """
    备份 model 中的所有参数并返回一个字典,
    方便后续在使用完 fast_weights 后恢复到原始参数.
    """
    backup = {}
    for name, param in model.named_parameters():
        backup[name] = param.data.clone()
    # 添加断言以确保所有参数已备份
    expected_params = set([name for name, _ in model.named_parameters()])
    backup_params = set(backup.keys())
    assert expected_params == backup_params, "Some parameters are missing in the backup."
    return backup


def restore_model_params(model, backup):
    """
    将 model 的所有参数恢复为 backup 中的副本.
    backup 即上述 backup_model_params 的返回值
    """
    for name, param in model.named_parameters():
        if name in backup:
            param.data.copy_(backup[name])
        else:
            raise KeyError(f"Parameter '{name}' not found in backup.")

# MAML (Meta-learning)
class MAML:
    def __init__(self, model, configs):
        self.model = model
        self.configs = configs
        self.lr_inner = configs.inner_lr_init
        # 外层优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=configs.outer_lr_max, weight_decay=configs.weight_decay)

        # 外层学习率调度器
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=configs.outer_lr_max,
            epochs=configs.mm_num_epochs,
            steps_per_epoch=1,
            pct_start=configs.warmup_epochs/configs.mm_num_epochs,
            anneal_strategy='cos',
            final_div_factor=configs.outer_lr_max/configs.outer_lr_min,
            last_epoch=-1
        )

        # 内层学习率调度器
        dummy_optimizer = optim.SGD([torch.tensor(self.lr_inner, requires_grad=False)], lr=self.lr_inner)
        self.inner_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            dummy_optimizer,
            T_0=configs.warmup_epochs,
            T_mult=2,
            eta_min=configs.inner_lr_min,
            last_epoch=-1
        )
    
    def adapt(self, support_x, support_y, loss_fn):
        # 1. 获取当前元参数的 state_dict 副本
        #    这是内层适应的起点。
        meta_params_state_dict = {name: param.clone() for name, param in self.model.named_parameters()}

        # 2. 创建一个用于任务适应的参数副本，后续更新将在此副本上进行
        adapted_state_dict = {name: param.clone() for name, param in meta_params_state_dict.items()}

        # 3. 获取当前用于内层适应的学习率
        #    self.lr_inner 在 meta_update 中根据 inner_scheduler 更新
        current_inner_lr = self.lr_inner

        # 4. 循环执行 `inner_steps` 次内层梯度更新
        for step in range(self.configs.inner_steps):
            # --- 内循环步骤逻辑 ---

            # 使用 forward_with_params 函数，它接受参数字典并避免直接修改模型实例的状态
            # 这样可以保持模型实例的原始状态，只操作参数的副本
            # forward_with_params(model, src, adapted_params)
            preds = forward_with_params(self.model, support_x, adapted_state_dict)

            # 计算支持集上的损失
            task_loss = loss_fn(preds, support_y)

            # 计算损失相对于当前任务参数（adapted_state_dict 中的参数）的梯度
            # `create_graph=True` 对于 MAML 至关重要，允许梯度反向传播通过适应过程
            
            # 获取 adapted_state_dict 中所有需要梯度的参数列表
            # 假设模型的所有参数都会被适应，所以我们从 state_dict 中提取它们
            params_to_update = [p for p in adapted_state_dict.values() if p.requires_grad]
            
            # 计算梯度
            grads = torch.autograd.grad(task_loss, params_to_update, create_graph=True, allow_unused=True)

            # 使用计算出的梯度和内层学习率更新 adapted_state_dict 中的参数
            new_adapted_state_dict = {}
            grad_iterator = iter(grads) # 迭代器用于按顺序获取梯度
            for name, param in adapted_state_dict.items():
                if param.requires_grad:
                    grad = next(grad_iterator)
                    if grad is not None:
                        new_adapted_state_dict[name] = param - current_inner_lr * grad
                    else:
                        new_adapted_state_dict[name] = param
                else:
                    # 非可训练参数保持不变
                    new_adapted_state_dict[name] = param
            
            # 将更新后的参数字典赋值回 adapted_state_dict，供下一轮迭代使用
            adapted_state_dict = new_adapted_state_dict

        # 5. 循环结束后，`adapted_state_dict` 包含了经过 `inner_steps` 次更新后的参数
        #    返回这个字典，供外层循环（meta_update）使用。
        return adapted_state_dict

    def meta_update(self, meta_loss):
        """
        执行外层元参数更新。
        """
        self.optimizer.zero_grad()
        meta_loss.backward()
        # 梯度裁剪有助于稳定训练
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.configs.mm_max_norn)
        self.optimizer.step()
        
        # 更新内外层学习率
        self.scheduler.step()
        # 更新 self.lr_inner 以供下一个 epoch 的适应使用
        self.inner_scheduler.step()
        self.lr_inner = self.inner_scheduler.optimizer.param_groups[0]['lr']
        
def _evaluate_on_global_val(model, val_loader, loss_function, eval_metrics, configs, device):
    model.eval()
    total_loss, total_mape, count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            # 更新模型调用
            preds = model.forward(batch_inputs) # [T15](10)
            loss = loss_function(preds, batch_targets)
            total_loss += loss.item()
            mape = eval_metrics(batch_targets, preds)
            total_mape += mape.item()
            count += 1
    avg_loss = total_loss / (count if count > 0 else 1)
    avg_mape = total_mape / (count if count > 0 else 1)
    return avg_loss, avg_mape

def evaluate_model(model, data_loader, device):
    """
    评估模型在给定数据集上的性能
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model.forward_autoregressive(data)
            loss = loss_function(output, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f"Evaluation - Average Loss: {avg_loss:.4f}")
    return avg_loss

def load_mmformer_model(configs):
    """加载已训练的MMformer模型"""
    device = configs.device
    model = MMformer(configs).to(device)
    checkpoint = torch.load(configs.save_path, map_location=device)
    model.load_state_dict(torch.load(configs.save_path, map_location=device))
    model.eval()  # 设置为评估模式
    print(f"MMformer model loaded from {configs.save_path}")
    return model

# Define Evaluation Metrics
def eval_metrics(actual, predicted, epsilon=configs.epsilon):
    filtered_actual = actual
    filtered_predicted = predicted
    mape = torch.mean(torch.abs((filtered_actual - filtered_predicted) / (torch.abs(filtered_actual) + epsilon))) * 100
    return mape

# MMformer Training
def train_mmformer(data, configs):   
    # Customize how much data to use for predictions
    forecast_length = data.shape[1]
    data_forecast = data[:, :forecast_length, :]

    # Divide the dataset
    train_days = 365 * 2
    val_days = 365 * 2 + 366
    
    # StandardScaler for train
    scaler = StandardScaler()
    train_data_raw = data_forecast[:, :train_days, :]
    scaler.fit(train_data_raw.reshape(-1, data_forecast.shape[-1]))
    
    # Transform for all
    scaled_data = scaler.transform(data_forecast.reshape(-1, data_forecast.shape[-1])).reshape(data_forecast.shape)
    
    # # Divide the dataset
    train_data = scaled_data[:, :train_days, :]
    val_data = scaled_data[:, train_days:val_days, :]
    test_data = scaled_data[:, val_days:, :]
    
    joblib.dump(scaler, 'mmformer_scaler_meta.pkl')  # 保存scaler
    print(scaled_data.shape)
    
    # Create datasets
    def create_city_based_tasks(data_3d, seq_len, pred_len):
        """
        为每个城市创建一个独立的TensorDataset（任务）。
        返回一个任务列表，其中每个元素都是一个城市的 TensorDataset。
        """
        tasks = []
        num_cities = data_3d.shape[0]
        for city_idx in range(num_cities):
            city_data = data_3d[city_idx] # 获取单个城市的数据 (days, features)
            num_days, num_features = city_data.shape
            num_windows = num_days - seq_len - pred_len + 1
    
            if num_windows <= 0:
                # 如果某个城市的数据不足以创建至少一个窗口，则跳过
                print(f"Warning: City {city_idx} has insufficient data ({num_days} days) for seq_len={seq_len} and pred_len={pred_len}. Skipping this task.")
                continue
            
            sequences_x = []
            sequences_y = []
            for i in range(num_windows):
                start_index = i
                end_index = i + seq_len + pred_len
                window = city_data[start_index:end_index, :]
                sequences_x.append(window[:seq_len, :])
                sequences_y.append(window[seq_len:, :])
            
            sequences_tensor = torch.from_numpy(np.array(sequences_x)).float()
            targets_tensor = torch.from_numpy(np.array(sequences_y)).float()
            
            tasks.append(TensorDataset(sequences_tensor, targets_tensor))
            
        return tasks
    
    def create_global_sequences(data_3d, seq_len, pred_len):
        """
        从所有城市数据中创建统一的、全局的序列和目标张量。
        返回两个张量：(所有窗口, seq_len, features) 和 (所有窗口, pred_len, features)
        """
        all_sequences_x = []
        all_sequences_y = []
        num_cities = data_3d.shape[0]
    
        for city_idx in range(num_cities):
            city_data = data_3d[city_idx]
            num_days = city_data.shape[0]
            num_windows = num_days - seq_len - pred_len + 1
    
            if num_windows <= 0:
                continue  # 跳过数据不足的城市
    
            for i in range(num_windows):
                window = city_data[i : i + seq_len + pred_len, :]
                all_sequences_x.append(window[:seq_len, :])
                all_sequences_y.append(window[seq_len:, :])
    
        # 将所有城市的窗口数据合并成两个大张量
        sequences_tensor = torch.from_numpy(np.array(all_sequences_x)).float()
        targets_tensor = torch.from_numpy(np.array(all_sequences_y)).float()
    
        return sequences_tensor, targets_tensor

    # 使用新函数为训练集创建任务列表
    print("Creating city-based tasks for training...")
    train_tasks = create_city_based_tasks(train_data, configs.mm_seq_len, configs.mm_pred_len)
    print(f"Successfully created {len(train_tasks)} training tasks (one per city).")
    
    # 全局验证集和测试集保持不变，用于评估元模型性能
    print("Creating global validation and test datasets...")
    # 使用新函数为验证集和测试集创建全局数据集
    val_sequences, val_targets = create_global_sequences(val_data, configs.mm_seq_len, configs.mm_pred_len)
    test_sequences, test_targets = create_global_sequences(test_data, configs.mm_seq_len, configs.mm_pred_len)
    
    val_dataset = TensorDataset(val_sequences, val_targets)
    test_dataset = TensorDataset(test_sequences, test_targets)
    
    global_val_loader = DataLoader(val_dataset, batch_size=configs.batch_size_mmformer, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size_mmformer, pin_memory=True)

    # Initialize the model and optimizer
    device = configs.device
    model = MMformer(configs).to(device)
    maml = MAML(model, configs)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 检查是否需要加载模型
    checkpoint_path = configs.save_path
    if configs.load_mmformer and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully.")

        # 检查是否是完整的检查点，用于恢复训练
        if configs.train_mmformer and 'optimizer_state_dict' in checkpoint:
            maml.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            maml.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            maml.inner_scheduler.load_state_dict(checkpoint['inner_scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            # 恢复内部学习率
            maml.lr_inner = maml.inner_scheduler.optimizer.param_groups[0]['lr']
            print(f"Optimizer, schedulers, and epoch restored. Resuming training from epoch {start_epoch}.")
        else:
            print("Warning: Checkpoint does not contain optimizer state, or training is disabled. Starting a new training session.")

        # 如果只是加载模型而不训练，则直接返回
        if not configs.train_mmformer:
            model.eval()
            print("Model loaded in evaluation mode (no training).")
            return model, test_loader, scaler

    if configs.train_mmformer:
        trigger_times = 0
        print(f"Starting MAML training from epoch {start_epoch}")
        
        # 修训练循环从 start_epoch 开始
        for epoch in range(start_epoch, configs.mm_num_epochs):
            model.train()
            
            # --- MAML 内外循环逻辑保持不变 ---
            num_available_tasks = len(train_tasks)
            meta_batch_size = min(configs.meta_batch_size, num_available_tasks)
            if meta_batch_size < configs.meta_batch_size:
                 print(f"Warning: Using {meta_batch_size} tasks instead of {configs.meta_batch_size}.")
            
            task_indices = np.random.choice(num_available_tasks, size=meta_batch_size, replace=False)
            meta_loss_accumulator = []
            
            for task_idx in task_indices:
                task_dataset = train_tasks[task_idx]
                if len(task_dataset) < 2: continue
                support_size = len(task_dataset) // 2
                query_size = len(task_dataset) - support_size
                if support_size == 0 or query_size == 0: continue
                support_set, query_set = random_split(task_dataset, [support_size, query_size])
                support_loader = DataLoader(support_set, batch_size=min(configs.inner_batch_size, len(support_set)), shuffle=True)
                query_loader = DataLoader(query_set, batch_size=min(configs.inner_batch_size, len(query_set)), shuffle=True)
                
                try:
                    support_x, support_y = next(iter(support_loader))
                    support_x, support_y = support_x.to(device), support_y.to(device)
                except StopIteration:
                    continue
                
                support_preds = model(support_x)
                support_loss = loss_function(support_preds, support_y)
                adapted_params = maml.adapt(support_x, support_y, loss_function)
                
                try:
                    query_x, query_y = next(iter(query_loader))
                    query_x, query_y = query_x.to(device), query_y.to(device)
                except StopIteration:
                    continue
                
                query_preds = forward_with_params(model, query_x, adapted_params)
                query_loss = loss_function(query_preds, query_y)
                meta_loss_accumulator.append(query_loss)

            if not meta_loss_accumulator:
                print(f"[Epoch {epoch+1}/{configs.mm_num_epochs}] No valid tasks for meta-update. Skipping.")
                continue
                
            meta_loss = torch.stack(meta_loss_accumulator).mean()
            maml.meta_update(meta_loss)
            
            val_loss_global, val_mape_global = _evaluate_on_global_val(model, global_val_loader, loss_function, eval_metrics, configs, device)
            
            outer_lr = maml.optimizer.param_groups[0]['lr']
            inner_lr = maml.lr_inner
            print(f"[Epoch {epoch+1}/{configs.mm_num_epochs}] meta_loss={meta_loss.item():.6f}, "
                  f"global_val_loss={val_loss_global:.6f}, global_val_mape={val_mape_global:.2f}%, "
                  f"outer_lr={outer_lr:.2e}, inner_lr={inner_lr:.2e}")
            
            # --- 保存检查点逻辑 ---
            if val_loss_global < best_val_loss:
                best_val_loss = val_loss_global
                trigger_times = 0
                
                # 保存完整的检查点字典
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': maml.optimizer.state_dict(),
                    'scheduler_state_dict': maml.scheduler.state_dict(),
                    'inner_scheduler_state_dict': maml.inner_scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, configs.save_path)
                print(f"==> Best val_loss={best_val_loss:.6f}, checkpoint saved to {configs.save_path}")
            else:
                trigger_times += 1
                if trigger_times >= configs.patience:
                    print(f'Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss}')
                    break
        
        print("Training finished. Loading best model...")
        # 训练结束后加载最佳模型权重
        checkpoint = torch.load(configs.save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, test_loader, scaler

# Load and Visualization
# MMformer Loaded
def load_mmformer_model(configs):
    device = configs.device
    model = MMformer(configs).to(device)
    print(f"MMformer model loaded from {configs.save_path}")
    return model

# MMformer to predict
def predict_with_mmformer(mmformer_model, test_loader, scaler, configs, num_mc_samples=100):
    num_mc_samples = configs.num_mc_samples
    device = configs.device
    mmformer_model.to(device) 

    all_predictions_mc = []
    all_targets = []

    mmformer_model.eval()

    total_mse, total_mae, total_mape, batches_count = 0, 0, 0, 0
    
    # 迭代测试加载器中的数据
    for batch_inputs, batch_targets in test_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        batch_predictions_mc_samples = []

        # --- MC Dropout 采样过程 ---
        # 临时将模型设置为训练模式，以激活 Dropout 层
        mmformer_model.train()

        # 在 no_grad 上下文中执行多次前向传播
        with torch.no_grad():
            for _ in range(num_mc_samples):
                # 执行前向传播（此时 Dropout 激活）
                preds = mmformer_model.forward(batch_inputs)
                batch_predictions_mc_samples.append(preds.cpu())

        # 收集完当前批次的 MC 样本后，切回评估模式
        mmformer_model.eval()

        # 将当前批次的预测堆叠起来，形状: (num_mc_samples, batch_size, pred_len, feature_size)
        batch_predictions_mc_tensor = torch.stack(batch_predictions_mc_samples, dim=0)

        # --- 计算当前批次的平均预测 ---
        # 形状: (batch_size, pred_len, feature_size)
        batch_mean_predictions = torch.mean(batch_predictions_mc_tensor, dim=0)
        batch_targets_cpu = batch_targets.cpu()

        # --- 计算并累加标准指标 ---
        # 使用 batch_mean_predictions 计算 MSE, MAE, MAPE，以提供一个整体性能的基准
        mse_val = F.mse_loss(batch_mean_predictions, batch_targets_cpu).item()
        mae_val = F.l1_loss(batch_mean_predictions, batch_targets_cpu).item()
        mape_val = eval_metrics(batch_targets_cpu, batch_mean_predictions).item() 
        
        total_mse += mse_val
        total_mae += mae_val
        total_mape += mape_val
        batches_count += 1
        
        # --- 收集用于最终计算整体方差和返回的张量 ---
        all_predictions_mc.append(batch_predictions_mc_tensor) # 收集所有 MC 样本预测
        all_targets.append(batch_targets.cpu())                # 收集真实目标值

    # --- 最终指标计算和打印 ---
    # 确保 batches_count > 0 避免除零错误
    avg_mse = total_mse / batches_count if batches_count > 0 else 0
    avg_mae = total_mae / batches_count if batches_count > 0 else 0
    avg_mape = total_mape / batches_count if batches_count > 0 else 0
    print(f"[MMformer Test] MSE={avg_mse:.5f}, MAE={avg_mae:.5f}, MAPE={avg_mape:.5f}%")

    # --- 拼接所有批次的 MC 样本预测和目标值 ---
    # 最终形状: (total_num_mc_samples, total_samples, pred_len, feature_size)
    all_predictions_mc_tensor = torch.cat(all_predictions_mc, dim=1) 
    all_targets_tensor = torch.cat(all_targets, dim=0)            

    # --- 计算最终的平均预测和方差 ---
    # 形状: (total_samples, pred_len, feature_size)
    final_mean_predictions = torch.mean(all_predictions_mc_tensor, dim=0)
    final_prediction_variance = torch.var(all_predictions_mc_tensor, dim=0) 

    # --- 保存预测结果 ---
    # 保存最终的平均预测结果，以覆盖或替代原 predict_with_mmformer 的保存行为
    np.save(configs.prediction_path, final_mean_predictions.numpy())
    print(f"Mean predictions saved to {configs.prediction_path}, shape: {final_mean_predictions.shape}")
    
    print(f"MC Dropout samples collected: {num_mc_samples} per data point.")
    print(f"Final Mean prediction shape: {final_mean_predictions.shape}")
    print(f"Final Prediction variance shape: {final_prediction_variance.shape}")
    
    # 返回平均预测、方差和所有真实目标值
    return final_mean_predictions, final_prediction_variance, all_targets_tensor

def reaggregate_predictions(predictions, num_cities, configs):
    # 确保 predictions 是 torch.Tensor 类型
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions).float()
    
    pred_len = configs.mm_pred_len
    feature_size = predictions.shape[-1]
    
    # 确认窗口数量计算正确
    num_windows_per_city = predictions.shape[0] // num_cities
    if num_windows_per_city < 1:
        print(f"Warning: predictions shape {predictions.shape}，num_cities={num_cities}")
        # 如果预测形状与城市数不匹配，直接返回原始预测
        return predictions
        
    # 重新聚合预测结果
    reaggregated_preds = torch.zeros((num_cities, pred_len, feature_size), device=predictions.device)
    for city_idx in range(num_cities):
        city_preds = predictions[city_idx * num_windows_per_city: (city_idx + 1) * num_windows_per_city]
        reaggregated_preds[city_idx] = torch.mean(city_preds, dim=0)
    
    return reaggregated_preds

# MMformer visualization
def visualize_mmformer_predictions(mmformer_model, test_loader, scaler, configs, device, sample=0):
    """
    Visualize MMformer's predictions and actual values。
    """
    mmformer_model.eval()
    device = configs.device
    # Get sample data in a batch and make predictions
    test_inputs, test_targets = next(iter(test_loader))
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    
    # Predictions from the model
    with torch.no_grad():
        predictions = mmformer_model.forward(test_inputs)
    
    # Move data from the GPU to the CPU and convert to numpy arrays for plotting with Matplotlib
    test_inputs = test_inputs.cpu().numpy()
    test_targets = test_targets.cpu().numpy()
    test_outputs = predictions.cpu().numpy()
    
    np.save('MMformer_preds.npy', test_outputs)
    np.save('MMformer_targets.npy', test_targets)
    np.save('MMformer_inputs.npy', test_inputs)
    
    # 使用传入的scaler进行逆标准化
    test_inputs_inv = scaler.inverse_transform(test_inputs.reshape(-1, configs.mm_feature_size)).reshape(test_inputs.shape)
    test_targets_inv = scaler.inverse_transform(test_targets.reshape(-1, configs.mm_feature_size)).reshape(test_targets.shape)
    test_outputs_inv = scaler.inverse_transform(test_outputs.reshape(-1, configs.mm_feature_size)).reshape(test_outputs.shape)
    
    # 为调试添加打印语句，检查值域（运行后查看控制台输出）
    print("原始输入 min/max:", test_inputs.min(), test_inputs.max())
    print("逆标准化输入 min/max:", test_inputs_inv.min(), test_inputs_inv.max())
    print("原始预测 min/max:", test_outputs.min(), test_outputs.max())
    print("逆标准化预测 min/max:", test_outputs_inv.min(), test_outputs_inv.max())

    # Draw a graph (you can choose a specific sample to draw)
    sample_index = 0  # select the sample
    input_seq_len = configs.mm_seq_len
    pred_len = configs.mm_pred_len
    
    plt.figure(figsize=(15, 5))
    
    # Plotting the input sequence
    plt.plot(range(input_seq_len), test_inputs_inv[sample_index, :, 0], label='Input Sequence')
    
    # Draw the true target sequence (the target sequence starts after input_seq_len)
    plt.plot(range(input_seq_len, input_seq_len + pred_len), test_targets_inv[sample_index, :, 0], label='True Target')
    
    # Draw the predicted target sequence (the predicted sequence starts after input_seq_len)
    plt.plot(range(input_seq_len, input_seq_len + pred_len), test_outputs_inv[sample_index, :, 0], label='Predicted Target')
    
    plt.title('MMformer Prediction for Air Quality Dataset')
    plt.xlabel('Time Steps')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Best model estimation
    mmformer_model.eval()
    
    # Obtain data from the test loader
    test_iter = iter(test_loader)
    inputs, true_values = next(test_iter)
    inputs, true_values = inputs.to(device), true_values.to(device)
    
    # Best model prediction
    with torch.no_grad():
        # 更新模型调用
        predictions = mmformer_model.forward(inputs)
    
    # 之后进行可视化
    inputs_np = inputs.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    true_values_np = true_values.cpu().numpy()
    
    # Inverse transform the standardized data
    true_values_np = scaler.inverse_transform(true_values.cpu().numpy().reshape(-1, configs.mm_feature_size)).reshape(true_values.shape)
    predictions_np = scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, configs.mm_feature_size)).reshape(predictions.shape)
    
    # Samples prediction MSE
    mse_list = []
    for i in range(true_values_np.shape[0]):
        mse = ((true_values_np[i] - predictions_np[i]) ** 2).mean()
        mse_list.append(mse)
    
    # Convert mse_list to NumPy array
    mse_array = np.array(mse_list)
    
    # Select the indices of the 5 samples with the lowest MSE
    num_samples = 5
    best_sample_indices = np.argsort(mse_array)[:num_samples]
    
    # Sort the best sample indices based on MSE in descending order
    sorted_sample_indices = best_sample_indices[np.argsort(mse_array[best_sample_indices])[::-1]]
    
    # Select features
    num_features = true_values_np.shape[2]
    
    # Plot
    if num_features == 1:
        # Create a single column subplot grid, axes will be a one-dimensional array
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 10))
        # Make sure axes is a numpy array so that it can be processed uniformly (even if there is only one Axes object)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
    else:
        # When num_features > 1, create a multi-column subgraph grid, axes is a two-dimensional array
        fig, axes = plt.subplots(num_samples, num_features, figsize=(15, 10))
    
    for i, sample_idx in enumerate(sorted_sample_indices):
        for j in range(num_features):
            if num_features == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]
            ax.plot(true_values_np[sample_idx, :, j], label='True Values' if i == 0 and j == 0 else None)
            ax.plot(predictions_np[sample_idx, :, j], label='Predictions' if i == 0 and j == 0 else None, linestyle='--')
            if i == 0:
                ax.set_title(f'Feature {j}')
            if j == 0:
                ax.set_ylabel(f'Sample {i+1}')
    
    # Add legend outside the subplots
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, borderaxespad=0.1)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the spacing and positioning of the subplots
    plt.show()

def main():
    configs = Config()
    if configs.visualize_predictions:
        os.makedirs(configs.visualization_output_dir, exist_ok=True)
        
    data_path = os.path.join(configs.data_dir, configs.data_path)
    if not os.path.exists(data_path):
        print(f"文件 {data_path} 不存在，请检查路径是否正确")
        return
    data = np.load(data_path)
    print(f'数据形状: {data.shape}')
    
    mmformer_model = None
    test_loader = None
    scaler = None
    mmformer_available = False
    
    if configs.train_mmformer or (configs.load_mmformer and os.path.exists(configs.save_path)):
        mmformer_model, test_loader, scaler = train_mmformer(data, configs)
        mmformer_available = True
    else:
        print("MMformer 未激活，跳过相关操作。")
        
    # 只有在MMformer可用时才执行预测和可视化
    if mmformer_available and scaler is not None:
        # MMformer Prediction
        # 正确解包2个返回值
        mean_preds, pred_variance, targets_tensor = predict_with_mmformer(mmformer_model, test_loader, scaler, 
                                                                          configs, num_mc_samples=configs.num_mc_samples)
        
        # --- 打印 final_prediction_variance 的统计信息 ---
        # 确保 pred_variance 在 CPU 上用于 NumPy 操作或 PyTorch 统计计算
        if pred_variance is not None:
            pred_variance_cpu = pred_variance.cpu()
    
            # 计算全局平均方差
            global_avg_variance = torch.mean(pred_variance_cpu)
            # 计算全局平均标准差
            global_avg_std_dev = torch.mean(torch.sqrt(pred_variance_cpu))
    
            # 查找最大和最小方差
            max_variance = torch.max(pred_variance_cpu)
            min_variance = torch.min(pred_variance_cpu)
    
            print("\n--- MMformer Final Prediction Variance Statistics ---")
            print(f"Global Average Variance: {global_avg_variance.item():.6f}")
            print(f"Global Average Standard Deviation: {global_avg_std_dev.item():.6f}")
            print(f"Maximum Variance across all samples/timesteps/features: {max_variance.item():.6f}")
            print(f"Minimum Variance across all samples/timesteps/features: {min_variance.item():.6f}")
            print("---------------------------------------------------\n")
        else:
            print("Warning: Final prediction variance is None. Cannot compute statistics.")
            
        # 计算每个样本的平均预测方差（跨时间步和特征）
        variance_per_sample = torch.mean(pred_variance, dim=(1, 2)) # 形状: (total_samples,)
        
        # 找到方差最大的样本索引
        max_variance_sample_idx = torch.argmax(variance_per_sample).item()
        print(f"Sample Index with Maximum Variance for Visualization: {max_variance_sample_idx}")
        
        # 重组预测结果回城市级别
        num_cities = data.shape[0]
        # 直接使用返回的predictions_tensor，无需torch.cat()
        reaggregated_preds = reaggregate_predictions(
            mean_preds, num_cities, configs
        )

        # MMformer Visualization prediction results
        visualize_mmformer_predictions(mmformer_model, test_loader, scaler, configs, configs.device, sample=0)
        
    else:
        print("MMformer or scaler not available. Skipping prediction and visualization to avoid inconsistencies.")

if __name__ == '__main__':
    main()
