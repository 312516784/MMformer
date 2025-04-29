import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import copy
import gc
import time

class Config:
    # Define model dimensions
    d_model = 768  # Embedding dimension
    n_heads = 8  # Number of attention heads
    d_ff = 1024  # Dimension of feed forward network
    e_layers = 8  # Number of encoder layers
    d_layers = 8  # Number of decoder layers
    seq_len = 30  # Sequence length (e.g., number of days for one city)
    pred_len = 10  # Prediction length (e.g., predict next 7 days)
    dropout = 0.2  # Dropout rate
    factor = 365  # ProbAttention factor
    feature_size = 2 # Feature size
    output_attention = True  # Whether to output attention weights
    training = True        # Whether to continue to train or re-train the model
    batch_size = 384          # Batch size
    num_epochs = 50         # number of epochs
    scheduled_sampling = True  
    sampling_ratio = 0.3       
    outer_lr_max = 0.00005     
    outer_lr_min = 1e-9    
    inner_lr_init = 1e-7     
    inner_lr_min = 1e-10     
    warmup_epochs = 2         
    lr_scheduler_patience = 3 
    weight_decay = 0.05       
    patience = 3         
    save_path = "best_Meta_MMformer_model.pth"
    
    epsilon = 1
    min_actual = 1e-1

configs = Config()

class MinMaxLoss(nn.Module):
    def __init__(self):
        super(MinMaxLoss, self).__init__()

    def forward(self, outputs, targets):
        min_loss = torch.min(outputs, targets).mean()
        max_loss = torch.max(outputs, targets).mean()
        return max_loss - min_loss

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, outputs, targets):
        return torch.mean(torch.abs((outputs - targets) / (targets + 1e-8))) * 100

loss_function = nn.MSELoss()

# Causal mask
def generate_causal_mask(seq_len, batch_size, n_heads, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    mask = mask.unsqueeze(0).unsqueeze(1) 
    mask = mask.expand(batch_size, n_heads, seq_len, seq_len)  
    return mask

# Monte Carlo Dropout
class MCDropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)
    
# Probabilistic Attention
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=7, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = MCDropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, H, L_q, D = queries.shape  # Batch size, Heads, Length, Depth per head
        L_k = keys.shape[2]

        # Scale dot-product attention
        scale = self.scale or 1. / (D ** 0.5)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, values)
        return context, attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(torch.float32)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape = (B, L, D)
        L = x.size(1)
        x = x + self.pe[:, :L, :].to(x.device)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model, bias=True)
        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x: (B, L, c_in)
        x = self.value_embedding(x)
        x = self.position_embedding(x)
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
        query = self.query_projection(query).view(B, L_q, self.n_heads, self.d_k)
        key   = self.key_projection(key).view(B, L_k, self.n_heads, self.d_k)
        value = self.value_projection(value).view(B, L_k, self.n_heads, self.d_v)
    
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
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return self.norm(src)

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, self_attention_layer, cross_attention_layer, d_model, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention_layer    # Self-attention layer
        self.cross_attention = cross_attention_layer  # Cross-attention layer for context from the encoder

        d_ff = d_ff or 4 * d_model
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.ELU(),
                                nn.Linear(d_ff, d_model))

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = MCDropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        out1, _ = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_mask)
        x = tgt + self.dropout(out1)
        x = self.norm1(x)
        
        out2, _ = self.cross_attention(x, memory, memory, attn_mask=memory_mask)
        x = x + self.dropout(out2)
        x = self.norm2(x)
        out3 = self.ff(x)
        x = x + self.dropout(out3)
        x = self.norm3(x)
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, layers, d_model):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)

# MMformer Model
class MMformer(nn.Module):
    def __init__(self, configs):
        super(MMformer, self).__init__()
        self.configs = configs
        # Embedding
        self.enc_embedding = DataEmbedding(configs.feature_size, configs.d_model, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.feature_size, configs.d_model, configs.dropout)
        
        # Encoder
        encoder_layer = EncoderLayer(
            AttentionLayer(
                ProbAttention(False, configs.factor, attention_dropout=configs.dropout),
                configs.d_model,
                configs.n_heads
            ),
            configs.d_model,
            configs.d_ff,
            dropout=configs.dropout
        )
        self.encoder = Encoder(encoder_layer, configs.e_layers, configs.d_model)
        
        # Decoder
        decoder_layers = [
            DecoderLayer(
                self_attention_layer=AttentionLayer(
                    ProbAttention(False, configs.factor, attention_dropout=configs.dropout),
                    configs.d_model,
                    configs.n_heads
                ),
                cross_attention_layer=AttentionLayer(
                    ProbAttention(False, configs.factor, attention_dropout=configs.dropout),
                    configs.d_model,
                    configs.n_heads
                ),
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                dropout=configs.dropout
            ) for _ in range(configs.d_layers)
        ]
        self.decoder = Decoder(decoder_layers, configs.d_model)
    
        self.projector = nn.Linear(configs.d_model, configs.feature_size, bias=True)
        self.pred_len = configs.pred_len
        
    def encode(self, x):
        enc_out = self.enc_embedding(x)     # (B, L, d_model)
        enc_out = self.encoder(enc_out)     # (B, L, d_model)
        return enc_out

    def decode_step(self, tgt_embedding, memory, tgt_mask=None, memory_mask=None):
        dec_out = self.decoder(tgt_embedding, memory, tgt_mask, memory_mask)  # (B, 1, d_model)
        out =  self.projector(dec_out)  # (B, 1, feature_size)
        return out
    
    def forward_autoregressive(self, src, tgt=None, pred_len=None, sampling_prob=1.0):
        if pred_len is None:
            pred_len = self.pred_len
        B, L, C = src.shape
        device = src.device
        
        memory = self.encode(src)    # => (B, L, d_model)
        n_heads = self.encoder.layers[0].attention_layer.n_heads
        first_step_input = torch.zeros((B, 1, C), device=device)
        dec_emb_list = [self.dec_embedding(first_step_input)] 
        
        outputs = []
        
        for step in range(pred_len):
            tgt_emb = torch.cat(dec_emb_list, dim=1)  # => (B, step+1, d_model)
            
            seq_len_dec = tgt_emb.size(1)
            mask = torch.tril(torch.ones(seq_len_dec, seq_len_dec, device=device)).bool()
            mask = mask.unsqueeze(0).unsqueeze(1) 
            mask = mask.expand(B, n_heads, seq_len_dec, seq_len_dec)  
            
            dec_out = self.decoder(tgt_emb, memory, tgt_mask=mask, memory_mask=None)  
            
            out_step = self.projector(dec_out[:, -1:, :]) 
            outputs.append(out_step)
        
            if self.training and (tgt is not None):
                use_real = torch.rand(B, device=device) < sampling_prob
                use_real = use_real.float().unsqueeze(1).unsqueeze(2)
                next_inp = use_real * tgt[:, step, :].unsqueeze(1) + \
                           (1 - use_real) * out_step
            else:
                next_inp = out_step
            
            dec_emb_list.append(self.dec_embedding(next_inp))
        
        outputs = torch.cat(outputs, dim=1)  # => (B, pred_len, C)
        return outputs
    
def forward_autoregressive_with_params(model, src, adapted_params, tgt=None, pred_len=None, sampling_prob=0.0):
    original_params_data = {}
    for name, param in model.named_parameters():
        original_params_data[name] = param.data.clone()
        
    try:
        for name, param in model.named_parameters():
            if name in adapted_params:
                param.data.copy_(adapted_params[name])
    
        outputs = model.forward_autoregressive(
            src, tgt=tgt, pred_len=pred_len, sampling_prob=sampling_prob)
    
    finally:
        for name, param in model.named_parameters():
            param.data.copy_(original_params_data[name])
    return outputs

def backup_model_params(model):
    backup = {}
    for name, param in model.named_parameters():
        backup[name] = param.data.clone()
    expected_params = set([name for name, _ in model.named_parameters()])
    backup_params = set(backup.keys())
    assert expected_params == backup_params, "Some parameters are missing in the backup."
    return backup


def restore_model_params(model, backup):
    for name, param in model.named_parameters():
        if name in backup:
            param.data.copy_(backup[name])
        else:
            raise KeyError(f"Parameter '{name}' not found in backup.")
    
# Meta-learning
class MAML:
    def __init__(self, model, configs):
        self.model = model
        self.lr_inner = configs.inner_lr_init
        self.optimizer = optim.AdamW(model.parameters(), lr=configs.outer_lr_max, weight_decay=configs.weight_decay) 
      
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=configs.outer_lr_max,
            epochs=configs.num_epochs,
            steps_per_epoch=1,
            pct_start=configs.warmup_epochs/configs.num_epochs,
            anneal_strategy='cos',
            final_div_factor=configs.outer_lr_max/configs.outer_lr_min
        )
        
        self.inner_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim.SGD([torch.tensor(self.lr_inner)], lr=self.lr_inner),
            T_0=configs.warmup_epochs,
            T_mult=2,
            eta_min=configs.inner_lr_min
        )

    def adapt(self, loss):
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        adapted_params = {
            name: param - self.lr_inner * grad
            for (name, param), grad in zip(self.model.named_parameters(), grads)
        }
        return adapted_params

    def meta_update(self, meta_loss):
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        
        self.scheduler.step()
        self.inner_scheduler.step()
        self.lr_inner = self.inner_scheduler.optimizer.param_groups[0]['lr']

# Define Evaluation Metrics
def eval_metrics(actual, predicted, epsilon=configs.epsilon, min_actual=configs.min_actual):
    mask = torch.abs(actual) > min_actual
    filtered_actual = actual[mask]
    filtered_predicted = predicted[mask]
    mape = torch.mean(torch.abs((filtered_actual - filtered_predicted) / (torch.abs(filtered_actual) + epsilon))) * 100
    return mape

# Create datasets
def create_sequences_3d(data_3d, seq_len, pred_len):
    num_cities, num_days, num_features = data_3d.shape
    num_windows = num_days - seq_len - pred_len + 1

    if num_windows <= 0:
        raise ValueError("Not enough days for given seq_len and pred_len")

    sequences_x = []
    sequences_y = []

    for city in range(num_cities):
        city_data = data_3d[city] 
        for i in range(num_windows):
            start_index = i
            end_index = i + seq_len + pred_len
            window = city_data[start_index:end_index, :]
            sequences_x.append(window[:seq_len, :])
            sequences_y.append(window[seq_len:, :])

    sequences_x = np.array(sequences_x)
    sequences_y = np.array(sequences_y)

    sequences_tensor = torch.from_numpy(sequences_x).float()
    targets_tensor = torch.from_numpy(sequences_y).float()

    print(f"num_cities: {num_cities}, num_days: {num_days}, num_features: {num_features}")
    print(f"seq_len: {seq_len}, pred_len: {pred_len}")
    print(f"Calculated num_windows: {num_windows}")
    print(f"sequences_x shape: {sequences_x.shape}")
    print(f"sequences_y shape: {sequences_y.shape}")
    print(f"sequences_tensor shape: {sequences_tensor.shape}")
    print(f"targets_tensor shape: {targets_tensor.shape}")

    return sequences_tensor, targets_tensor


# Calculagraph
start_time = time.time()
# Load your data
data = np.load('combined_precipitation_temperature_data.npy')

# Customize how much data to use for predictions
forecast_length = data.shape[1]
data_forecast = data[:, :forecast_length, :]

# StandardScaler for 3D data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_forecast.reshape(-1, data_forecast.shape[-1])).reshape(data_forecast.shape)

# Split the data into train, validation, and test sets
num_cities, num_days, num_features = scaled_data.shape
train_days = 160
val_days = 160+60

train_data = scaled_data[:, :train_days, :]
val_data = scaled_data[:, train_days:val_days, :]
test_data = scaled_data[:, val_days:, :]

# Convert data to PyTorch tensors
train_sequences, train_targets = create_sequences_3d(train_data, configs.seq_len, configs.pred_len)
val_sequences,   val_targets   = create_sequences_3d(val_data,   configs.seq_len, configs.pred_len)
test_sequences,  test_targets  = create_sequences_3d(test_data,  configs.seq_len, configs.pred_len)

train_dataset = TensorDataset(train_sequences, train_targets)
val_dataset   = TensorDataset(val_sequences,   val_targets)
test_dataset  = TensorDataset(test_sequences,  test_targets)

train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=configs.batch_size, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=configs.batch_size, pin_memory=True)

# Initialize model and optimizer
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MMformer(configs).to(device)
maml = MAML(model, configs)

# Check if we are continuing training from a saved model
if configs.training:
    model.load_state_dict(torch.load('best_Meta_MMformer_model.pth'))

model = model.to(device)

# Split the dataset into several different tasks
num_tasks = 1
task_size = len(train_dataset) // num_tasks
task_datasets = [torch.utils.data.Subset(train_dataset, range(i*task_size, (i+1)*task_size)) for i in range(num_tasks)]
pred_len = configs.pred_len
seq_len = configs.seq_len

def _evaluate_on_global_val(model, val_loader, loss_function, eval_metrics, configs, device):
    model.eval()
    total_loss = 0.0
    total_mape = 0.0
    count = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            batch_inputs  = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            preds = model.forward_autoregressive(
                src=batch_inputs, tgt=None, 
                pred_len=configs.pred_len,
                sampling_prob=0.0
            )
            loss = loss_function(preds, batch_targets)
            total_loss += loss.item()
    
            mape = eval_metrics(batch_targets, preds)
            total_mape += mape.item()
            count += 1
    
    avg_loss = total_loss / (count if count>0 else 1)
    avg_mape = total_mape / (count if count>0 else 1)
    return avg_loss, avg_mape

def meta_train(model, maml, tasks, configs, loss_function, eval_metrics, device, val_loader):
    num_epochs = configs.num_epochs
    best_val_loss = float('inf')
    best_model_state = None
    inner_steps = 3
    save_path = configs.save_path
    global_val_loader = val_loader 
    patience = configs.patience 
    trigger_times = 0 

    for epoch in range(num_epochs):
        model.train()
        epoch_meta_loss = 0.0

        for task in tasks:
            train_loader = task["train_loader"]
            train_iter = iter(train_loader)

            original_params = {name: param.clone().detach() for name, param in model.named_parameters()}
            adapted_params = {name: param.clone().detach().requires_grad_(True) for name, param in model.named_parameters()}

            for _ in range(inner_steps):
                try:
                    batch_inputs, batch_targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch_inputs, batch_targets = next(train_iter)

                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                preds = forward_autoregressive_with_params(
                    model, batch_inputs, adapted_params, tgt=batch_targets,
                    pred_len=configs.pred_len, sampling_prob=configs.sampling_ratio
                )
                loss = loss_function(preds, batch_targets)

                grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True, allow_unused=True)
                adapted_params = {
                    name: param - configs.inner_lr_init * (grad if grad is not None else torch.zeros_like(param))
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }

            try:
                val_inputs, val_targets = next(iter(val_loader))
            except StopIteration:
                val_inputs, val_targets = next(iter(val_loader))

            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)

            val_preds = forward_autoregressive_with_params(
                model, val_inputs, adapted_params, tgt=None,
                pred_len=configs.pred_len, sampling_prob=0.0
            )
            val_loss = loss_function(val_preds, val_targets)

            epoch_meta_loss += val_loss

            for name, param in model.named_parameters():
                param.data.copy_(original_params[name])

        avg_meta_loss = epoch_meta_loss / len(tasks)
        maml.optimizer.zero_grad()
        avg_meta_loss.backward()
        maml.optimizer.step()

        maml.scheduler.step()
        maml.inner_scheduler.step()

        val_loss_global, val_mape_global = _evaluate_on_global_val(
            model, global_val_loader, loss_function, eval_metrics, configs, device
        )

        print(f"[Epoch {epoch+1}/{num_epochs}] meta_loss={avg_meta_loss.item():.6f}, "
              f"global_val_loss={val_loss_global:.6f}, global_val_mape={val_mape_global:.2f}%")
        
        if val_loss_global < best_val_loss:
            best_val_loss = val_loss_global
            best_model_state = model.state_dict() 
            torch.save(best_model_state, save_path) 
            trigger_times = 0
            print(f"==> Best global_val_loss={best_val_loss:.6f}，Saved to {save_path}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}! Best val loss: {best_val_loss}')
                model.load_state_dict(best_model_state)
                break 
    return model

def test_mmformer_scaled(model,test_loader,device,loss_function,transform_metrics_fn,pred_len):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_mape = 0.0
    batches_count = 0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
    
            preds = model.forward_autoregressive(
                src=batch_inputs,
                pred_len=pred_len,  
                sampling_prob=0.0   
            )  
    
            B = batch_inputs.size(0)
            F_dim = batch_inputs.size(2) 
            preds_2d = preds.reshape(B * pred_len, F_dim).cpu().numpy()
            targets_2d = batch_targets.reshape(B * pred_len, F_dim).cpu().numpy()
    
            preds_torch = torch.tensor(preds_2d, dtype=torch.float32, device=device).reshape(B, pred_len, F_dim)
            targets_torch = torch.tensor(targets_2d, dtype=torch.float32, device=device).reshape(B, pred_len, F_dim)
    
            mse_val = F.mse_loss(preds_torch, targets_torch, reduction='mean').item()
            mae_val = F.l1_loss(preds_torch, targets_torch, reduction='mean').item()
    
            mape_val = transform_metrics_fn(targets_torch, preds_torch).item()
    
            total_mse += mse_val
            total_mae += mae_val
            total_mape += mape_val
            batches_count += 1
    
    avg_mse = total_mse / (batches_count if batches_count > 0 else 1)
    avg_mae = total_mae / (batches_count if batches_count > 0 else 1)
    avg_mape = total_mape / (batches_count if batches_count > 0 else 1)
    
    print(f"[MMformer Test] MSE={avg_mse:.5f}, MAE={avg_mae:.5f}, MAPE={avg_mape:.5f}%")
    return avg_mse, avg_mae, avg_mape

# Training loop
tasks = [{"train_loader": train_loader, "val_loader": val_loader}]
global_val_loader = val_loader 
model = meta_train(model, maml, tasks, configs, loss_function, eval_metrics, device, global_val_loader)

# Test Module
model = MMformer(configs)
model.load_state_dict(torch.load(configs.save_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

mse_val, mae_val, mape_val = test_mmformer_scaled(model=model,test_loader=test_loader,device=device,
                                                  loss_function=loss_function,transform_metrics_fn=eval_metrics,
                                                  pred_len=configs.pred_len)
print(f"Test Results: MSE={mse_val:.5f}, MAE={mae_val:.5f}, MAPE={mape_val:.5f}%")

best_model_state_dict = torch.load(configs.save_path)
model = MMformer(configs).to(device)
model.load_state_dict(best_model_state_dict)
model = model.to(device)
model.eval()

test_inputs, test_targets = next(iter(test_loader))
test_inputs = test_inputs.to(device)
test_targets = test_targets.to(device)

with torch.no_grad():
    src = test_inputs[:, :-pred_len, :]  
    tgt = test_inputs[:, -seq_len:-pred_len, :] 
    test_outputs = model.forward_autoregressive(
    src=src,
    tgt=tgt,
    pred_len=pred_len,
    sampling_prob=0.0)

test_inputs = test_inputs.cpu().numpy()
test_targets = test_targets.cpu().numpy()
test_outputs = test_outputs.cpu().numpy()

sample_index = 0 
input_seq_len = configs.seq_len
pred_len = configs.pred_len

plt.figure(figsize=(15, 5))
plt.plot(range(input_seq_len), test_inputs[sample_index, :, 0], label='Input Sequence')
plt.plot(range(input_seq_len, input_seq_len + pred_len), test_targets[sample_index, :, 0], label='True Target')
plt.plot(range(input_seq_len, input_seq_len + pred_len), test_outputs[sample_index, :, 0], label='Predicted Target')

plt.title('MMformer Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Features')
plt.legend()
plt.grid(True)
plt.show()

# Obtain data from the test loader
test_iter = iter(test_loader)
inputs, true_values = next(test_iter)
inputs, true_values = inputs.to(device), true_values.to(device)

with torch.no_grad():
    predictions = model.forward_autoregressive(
        src=inputs,
        pred_len=configs.pred_len,
        sampling_prob=0.0
    )

inputs_np = inputs.cpu().numpy()
predictions_np = predictions.cpu().numpy()
true_values_np = true_values.cpu().numpy()    

# Inverse transform the standardized data
inputs_np = scaler.inverse_transform(inputs.cpu().numpy().reshape(-1, num_features)).reshape(inputs.shape)
predictions_np = scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, num_features)).reshape(predictions.shape)

# Samples prediction MSE
mse_list = []
for i in range(inputs_np.shape[0]):
    mse = ((inputs_np[i, -configs.pred_len:, :] - predictions_np[i, -configs.pred_len:, :]) ** 2).mean()
    mse_list.append(mse)

# Convert mse_list to NumPy array
mse_array = np.array(mse_list)

# Select the indices of the 5 samples with the lowest MSE
num_samples = 5
best_sample_indices = np.argsort(mse_array)[:num_samples]

# Sort the best sample indices based on MSE in descending order
sorted_sample_indices = best_sample_indices[np.argsort(mse_array[best_sample_indices])[::-1]]

# Select features
num_features = inputs_np.shape[2]

# Plot
fig, axes = plt.subplots(num_samples, num_features, figsize=(15, 10))

for i, sample_idx in enumerate(sorted_sample_indices):
    for j in range(num_features):
        ax = axes[i, j]
        ax.plot(inputs_np[sample_idx, -configs.pred_len:, j], label='True Values' if i == 0 and j == 0 else None)
        ax.plot(predictions_np[sample_idx, -configs.pred_len:, j], label='Predictions' if i == 0 and j == 0 else None, linestyle='--')
        if i == 0:
            ax.set_title(f'Feature {j}')
        if j == 0:
            ax.set_ylabel(f'Sample {i+1}')

# Add legend outside the subplots
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, borderaxespad=0.1)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the spacing and positioning of the subplots
plt.show()
