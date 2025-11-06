"""
从零实现的Transformer模型 - MIDI音乐生成
不使用PyTorch/TensorFlow，手动实现所有组件
用于生成MIDI音乐序列
"""
import numpy as np
import math
from collections import Counter
import random
import urllib.request
import os
import zipfile
import pickle
from tqdm import tqdm
import mido
from mido import MidiFile, MidiTrack, Message


# ============================================================
# 1. 基础工具函数
# ============================================================

def softmax(x, axis=-1):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-6):
    """Layer Normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    return gamma * normalized + beta


def layer_norm_backward(dout, x, gamma, beta, eps=1e-6):
    """Layer Normalization反向传播"""
    # x: (batch_size, seq_len, d_model)
    # dout: (batch_size, seq_len, d_model)
    # gamma, beta: (d_model,)
    
    mean = np.mean(x, axis=-1, keepdims=True)  # (batch_size, seq_len, 1)
    var = np.var(x, axis=-1, keepdims=True)    # (batch_size, seq_len, 1)
    std = np.sqrt(var + eps)                    # (batch_size, seq_len, 1)
    normalized = (x - mean) / std              # (batch_size, seq_len, d_model)
    
    # 计算gamma和beta的梯度：需要沿着batch和seq_len维度求和
    # dgamma = sum over (batch, seq) of (dout * normalized)
    dgamma = np.sum(dout * normalized, axis=(0, 1))  # (d_model,)
    dbeta = np.sum(dout, axis=(0, 1))                # (d_model,)
    
    # 计算输入的梯度
    dnormalized = dout * gamma  # (batch_size, seq_len, d_model)
    dvar = np.sum(dnormalized * (x - mean) * -0.5 / (std ** 3), axis=-1, keepdims=True)
    dmean = np.sum(dnormalized * -1.0 / std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)
    dx = dnormalized / std + dvar * 2.0 * (x - mean) / x.shape[-1] + dmean / x.shape[-1]
    
    return dx, dgamma, dbeta


def softmax_backward(dout, x):
    """Softmax反向传播"""
    s = softmax(x, axis=-1)
    # Softmax的Jacobian矩阵是 s * (I - s^T)，但这里简化处理
    return dout * s * (1 - s)


def positional_encoding(seq_len, d_model):
    """位置编码（sinusoidal）"""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe


# ============================================================
# 2. Multi-Head Attention（手动实现Q、K、V）
# ============================================================

class MultiHeadAttention:
    """多头注意力机制 - 手动实现Q、K、V"""
    
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q、K、V的权重矩阵
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        
        # 输出投影矩阵
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, query, key, value, mask=None):
        """前向传播（保存中间结果用于反向传播）"""
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]
        
        # ========== 步骤1: 计算Q、K、V ==========
        Q = np.dot(query, self.W_q)  # (batch_size, seq_len_q, d_model)
        K = np.dot(key, self.W_k)     # (batch_size, seq_len_k, d_model)
        V = np.dot(value, self.W_v)   # (batch_size, seq_len_v, d_model)
        
        # ========== 步骤2: 重塑为多头 ==========
        Q = Q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len_v, self.num_heads, self.head_dim)
        
        # 转置: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # ========== 步骤3: 计算注意力分数 ==========
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        # ========== 步骤4: 应用mask ==========
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])
            scores = np.where(mask, -1e9, scores)
        
        # ========== 步骤5: Softmax ==========
        attention_weights = softmax(scores, axis=-1)
        
        # ========== 步骤6: 加权求和 ==========
        attention_output = np.matmul(attention_weights, V)
        
        # ========== 步骤7: 合并多头 ==========
        attention_output = attention_output.transpose(0, 2, 1, 3)
        attention_output = attention_output.reshape(batch_size, seq_len_q, self.d_model)
        
        # ========== 步骤8: 输出投影 ==========
        output = np.dot(attention_output, self.W_o)
        
        # 保存中间结果用于反向传播
        self.query = query
        self.key = key
        self.value = value
        self.Q = Q
        self.K = K
        self.V = V
        self.scores = scores
        self.attention_weights = attention_weights
        self.attention_output = attention_output
        self.mask = mask
        
        return output, attention_weights
    
    def backward(self, dout):
        """完整的反向传播"""
        batch_size, seq_len_q, d_model = dout.shape
        
        # ========== 步骤8反向: 输出投影 ==========
        dattention_output = np.dot(dout, self.W_o.T)  # (batch_size, seq_len_q, d_model)
        
        # ========== 步骤7反向: 合并多头 ==========
        dattention_output_reshaped = dattention_output.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim)
        dattention_output_transposed = dattention_output_reshaped.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len_q, head_dim)
        
        # ========== 步骤6反向: 加权求和 ==========
        dattention_weights = np.matmul(dattention_output_transposed, self.V.transpose(0, 1, 3, 2))  # (batch_size, num_heads, seq_len_q, seq_len_v)
        dV = np.matmul(self.attention_weights.transpose(0, 1, 3, 2), dattention_output_transposed)  # (batch_size, num_heads, seq_len_v, head_dim)
        
        # ========== 步骤5反向: Softmax ==========
        dscores = softmax_backward(dattention_weights, self.scores)
        
        # ========== 步骤4反向: Mask（不需要梯度） ==========
        # mask不影响梯度
        
        # ========== 步骤3反向: 注意力分数 ==========
        dscores = dscores / math.sqrt(self.head_dim)
        dQ = np.matmul(dscores, self.K)  # (batch_size, num_heads, seq_len_q, head_dim)
        dK = np.matmul(dscores.transpose(0, 1, 3, 2), self.Q)  # (batch_size, num_heads, seq_len_k, head_dim)
        
        # ========== 步骤2反向: 重塑 ==========
        dQ = dQ.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, d_model)
        dK = dK.transpose(0, 2, 1, 3).reshape(batch_size, self.key.shape[1], d_model)
        dV = dV.transpose(0, 2, 1, 3).reshape(batch_size, self.value.shape[1], d_model)
        
        # ========== 步骤1反向: Q、K、V计算 ==========
        dquery = np.dot(dQ, self.W_q.T)
        dkey = np.dot(dK, self.W_k.T)
        dvalue = np.dot(dV, self.W_v.T)
        
        # 计算权重梯度
        dW_q = np.dot(self.query.transpose(0, 2, 1).reshape(batch_size * d_model, seq_len_q), 
                     dQ.transpose(0, 2, 1).reshape(batch_size * d_model, seq_len_q).T)
        dW_q = np.mean(dW_q.reshape(batch_size, d_model, d_model), axis=0) if dW_q.ndim > 2 else dW_q
        
        dW_k = np.dot(self.key.transpose(0, 2, 1).reshape(batch_size * d_model, self.key.shape[1]), 
                     dK.transpose(0, 2, 1).reshape(batch_size * d_model, self.key.shape[1]).T)
        dW_k = np.mean(dW_k.reshape(batch_size, d_model, d_model), axis=0) if dW_k.ndim > 2 else dW_k
        
        dW_v = np.dot(self.value.transpose(0, 2, 1).reshape(batch_size * d_model, self.value.shape[1]), 
                     dV.transpose(0, 2, 1).reshape(batch_size * d_model, self.value.shape[1]).T)
        dW_v = np.mean(dW_v.reshape(batch_size, d_model, d_model), axis=0) if dW_v.ndim > 2 else dW_v
        
        # 简化：直接计算梯度
        dW_q = np.dot(self.query.reshape(-1, d_model).T, dQ.reshape(-1, d_model)) / batch_size
        dW_k = np.dot(self.key.reshape(-1, d_model).T, dK.reshape(-1, d_model)) / batch_size
        dW_v = np.dot(self.value.reshape(-1, d_model).T, dV.reshape(-1, d_model)) / batch_size
        dW_o = np.dot(self.attention_output.reshape(-1, d_model).T, dout.reshape(-1, d_model)) / batch_size
        
        return dquery, dkey, dvalue, dW_q, dW_k, dW_v, dW_o


# ============================================================
# 3. Feed Forward Network
# ============================================================

class FeedForward:
    """前馈神经网络"""
    
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        self.input = x  # 保存输入
        x = np.dot(x, self.W1) + self.b1
        self.pre_activation = x  # 保存ReLU前的值
        x = np.maximum(0, x)  # ReLU
        self.activation = x  # 保存ReLU后的值
        x = np.dot(x, self.W2) + self.b2
        return x
    
    def backward(self, dout):
        """完整的反向传播"""
        batch_size, seq_len, d_model = dout.shape
        
        # 第二层反向: dW2 = activation^T @ dout, db2 = sum(dout)
        dactivation = np.dot(dout, self.W2.T)  # (batch_size, seq_len, d_ff)
        dW2 = np.dot(self.activation.reshape(-1, self.activation.shape[-1]).T, 
                    dout.reshape(-1, dout.shape[-1])) / (batch_size * seq_len)
        db2 = np.sum(dout, axis=(0, 1))  # (d_model,)
        
        # ReLU反向: 只对>0的位置有梯度
        dpre_activation = dactivation * (self.pre_activation > 0)
        
        # 第一层反向: dW1 = input^T @ dpre_activation, db1 = sum(dpre_activation)
        dx = np.dot(dpre_activation, self.W1.T)  # (batch_size, seq_len, d_model)
        dW1 = np.dot(self.input.reshape(-1, self.input.shape[-1]).T, 
                    dpre_activation.reshape(-1, dpre_activation.shape[-1])) / (batch_size * seq_len)
        db1 = np.sum(dpre_activation, axis=(0, 1))  # (d_ff,)
        
        return dx, dW1, db1, dW2, db2


# ============================================================
# 4. Transformer Decoder Layer（用于自回归生成）
# ============================================================

class TransformerDecoderLayer:
    """Transformer解码器层（用于音乐生成）"""
    
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        # Layer Normalization参数
        self.norm1_gamma = np.ones(d_model)
        self.norm1_beta = np.zeros(d_model)
        self.norm2_gamma = np.ones(d_model)
        self.norm2_beta = np.zeros(d_model)
    
    def forward(self, x, mask=None):
        """前向传播（保存中间结果）"""
        self.input = x
        
        # Self-Attention + Residual + Layer Norm
        attn_output, _ = self.self_attn.forward(x, x, x, mask)
        self.attn_output = attn_output
        self.pre_norm1 = x + attn_output
        x = layer_norm(self.pre_norm1, self.norm1_gamma, self.norm1_beta)
        self.post_norm1 = x
        
        # Feed Forward + Residual + Layer Norm
        ff_output = self.feed_forward.forward(x)
        self.ff_output = ff_output
        self.pre_norm2 = x + ff_output
        x = layer_norm(self.pre_norm2, self.norm2_gamma, self.norm2_beta)
        self.post_norm2 = x
        
        return x
    
    def backward(self, dout, mask=None, optimizer=None):
        """完整的反向传播"""
        # ========== 第二层Layer Norm反向 ==========
        dx_norm2, dgamma2, dbeta2 = layer_norm_backward(dout, self.pre_norm2, self.norm2_gamma, self.norm2_beta)
        
        # ========== Feed Forward反向 ==========
        dff_output = dx_norm2  # Residual connection
        dx_ff, dW1, db1, dW2, db2 = self.feed_forward.backward(dff_output)
        
        # ========== 第一层Layer Norm反向 ==========
        dx_norm1 = dx_ff + dx_norm2  # Residual connection
        dx_pre_norm1, dgamma1, dbeta1 = layer_norm_backward(dx_norm1, self.pre_norm1, self.norm1_gamma, self.norm1_beta)
        
        # ========== Self-Attention反向 ==========
        dattn_output = dx_pre_norm1  # Residual connection
        dx_attn, _, _, dW_q, dW_k, dW_v, dW_o = self.self_attn.backward(dattn_output)
        
        # 合并输入梯度（因为Q、K、V都来自同一个输入x）
        dx = dx_attn + dx_attn + dx_attn  # 三个分支合并
        
        # 使用Adam优化器更新参数
        if optimizer is not None:
            optimizer.update(self.self_attn.W_q, dW_q, f"layer_attn_W_q")
            optimizer.update(self.self_attn.W_k, dW_k, f"layer_attn_W_k")
            optimizer.update(self.self_attn.W_v, dW_v, f"layer_attn_W_v")
            optimizer.update(self.self_attn.W_o, dW_o, f"layer_attn_W_o")
            
            optimizer.update(self.feed_forward.W1, dW1, f"layer_ff_W1")
            optimizer.update(self.feed_forward.b1, db1, f"layer_ff_b1")
            optimizer.update(self.feed_forward.W2, dW2, f"layer_ff_W2")
            optimizer.update(self.feed_forward.b2, db2, f"layer_ff_b2")
            
            optimizer.update(self.norm1_gamma, dgamma1, f"layer_norm1_gamma")
            optimizer.update(self.norm1_beta, dbeta1, f"layer_norm1_beta")
            optimizer.update(self.norm2_gamma, dgamma2, f"layer_norm2_gamma")
            optimizer.update(self.norm2_beta, dbeta2, f"layer_norm2_beta")
        else:
            # 如果没有优化器，直接更新（用于测试）
            self.self_attn.W_q -= 0.001 * dW_q
            self.self_attn.W_k -= 0.001 * dW_k
            self.self_attn.W_v -= 0.001 * dW_v
            self.self_attn.W_o -= 0.001 * dW_o
            
            self.feed_forward.W1 -= 0.001 * dW1
            self.feed_forward.b1 -= 0.001 * db1
            self.feed_forward.W2 -= 0.001 * dW2
            self.feed_forward.b2 -= 0.001 * db2
            
            self.norm1_gamma -= 0.001 * dgamma1
            self.norm1_beta -= 0.001 * dbeta1
            self.norm2_gamma -= 0.001 * dgamma2
            self.norm2_beta -= 0.001 * dbeta2
        
        return dx


# ============================================================
# 5. Adam优化器（手动实现）
# ============================================================

class AdamOptimizer:
    """Adam优化器（手动实现）"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # 时间步
        
        # 存储每个参数的momentum和variance
        self.m = {}  # momentum
        self.v = {}  # variance
    
    def get_state_key(self, param_name, param_id):
        """生成参数状态的唯一键"""
        return f"{param_name}_{id(param_id)}"
    
    def update(self, param, grad, param_name="param"):
        """使用Adam更新参数"""
        self.t += 1
        
        # 获取或初始化状态
        state_key = self.get_state_key(param_name, param)
        if state_key not in self.m:
            self.m[state_key] = np.zeros_like(param)
            self.v[state_key] = np.zeros_like(param)
        
        m = self.m[state_key]
        v = self.v[state_key]
        
        # Adam更新规则
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        
        # 偏差修正
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        
        # 更新参数
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # 保存状态
        self.m[state_key] = m
        self.v[state_key] = v


# ============================================================
# 6. Music Transformer模型
# ============================================================

class MusicTransformer:
    """音乐生成Transformer模型（GPT风格，只有Decoder）"""
    
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, d_ff=1024, max_len=512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # Embedding层
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # 位置编码
        self.pos_encoding = positional_encoding(max_len, d_model)
        
        # Decoder层（用于自回归生成）
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # 输出层
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02
        self.output_bias = np.zeros(vocab_size)
        
        # Adam优化器（为模型创建优化器）
        self.optimizer = None  # 将在训练时初始化
    
    def embed(self, token_ids):
        """词嵌入 + 位置编码"""
        batch_size, seq_len = token_ids.shape
        embedded = self.embedding[token_ids]  # (batch_size, seq_len, d_model)
        pos_enc = self.pos_encoding[:seq_len]  # (seq_len, d_model)
        return embedded + pos_enc
    
    def forward(self, token_ids, mask=None):
        """前向传播"""
        x = self.embed(token_ids)
        for layer in self.decoder_layers:
            x = layer.forward(x, mask)
        output = np.dot(x, self.output_proj) + self.output_bias
        return output
    
    def generate_causal_mask(self, seq_len):
        """生成causal mask"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        return mask


# ============================================================
# 6. MIDI数据处理
# ============================================================

def midi_to_sequence(midi_file_path, ticks_per_beat=480, time_quantization=120):
    """将MIDI文件转换为事件序列（包含音符、力度、时间、持续时间）"""
    try:
        mid = MidiFile(midi_file_path)
        ticks_per_beat = mid.ticks_per_beat
        
        # 存储所有事件：每个事件是 (note, velocity, time_offset, duration)
        events = []
        active_notes = {}  # {note: (start_time, velocity)}
        current_time = 0
        
        # 遍历所有track
        for track in mid.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    note = msg.note
                    velocity = msg.velocity
                    if 0 <= note <= 127:
                        # 记录note_on事件
                        active_notes[note] = (track_time, velocity)
                
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    note = msg.note
                    if note in active_notes:
                        start_time, velocity = active_notes[note]
                        duration = track_time - start_time
                        
                        # 量化时间偏移（相对于上一个事件）
                        time_offset = start_time - current_time
                        time_offset_quantized = max(0, int(time_offset / time_quantization))
                        
                        # 量化持续时间
                        duration_quantized = max(1, int(duration / time_quantization))
                        
                        # 限制范围以避免词表过大
                        time_offset_quantized = min(time_offset_quantized, 127)
                        duration_quantized = min(duration_quantized, 127)
                        
                        # 添加事件：(note, velocity, time_offset, duration)
                        events.append((note, velocity, time_offset_quantized, duration_quantized))
                        current_time = start_time
                        del active_notes[note]
        
        # 如果序列太短或太长，进行过滤
        if len(events) < 10:
            return None
        if len(events) > 1000:  # 减少最大长度，因为每个事件包含更多信息
            events = events[:1000]
        
        return events
    except Exception as e:
        # print(f"处理MIDI文件失败 {midi_file_path}: {e}")
        return None


def build_midi_vocab(sequences, max_vocab_size=512):
    """构建MIDI词表（包含音符、力度、时间偏移、持续时间）"""
    vocab = {}
    id_to_token = {}
    
    # 特殊标记
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    for i, token in enumerate(special_tokens):
        vocab[token] = i
        id_to_token[i] = token
    
    # 事件类型标记
    event_types = ['<note>', '<velocity>', '<time>', '<duration>']
    for event_type in event_types:
        idx = len(vocab)
        vocab[event_type] = idx
        id_to_token[idx] = event_type
    
    # MIDI音符 (0-127)
    for note in range(128):
        idx = len(vocab)
        vocab[f'note_{note}'] = idx
        id_to_token[idx] = f'note_{note}'
    
    # 力度 (0-127，但通常只用常见值)
    # 使用量化：只保留常见的力度值
    velocity_levels = [0, 32, 48, 64, 80, 96, 112, 127]  # 8个力度级别
    for vel in velocity_levels:
        idx = len(vocab)
        vocab[f'vel_{vel}'] = idx
        id_to_token[idx] = f'vel_{vel}'
    
    # 时间偏移 (0-127，已量化)
    for time_offset in range(128):
        idx = len(vocab)
        vocab[f'time_{time_offset}'] = idx
        id_to_token[idx] = f'time_{time_offset}'
    
    # 持续时间 (1-127，已量化)
    for duration in range(1, 128):
        idx = len(vocab)
        vocab[f'dur_{duration}'] = idx
        id_to_token[idx] = f'dur_{duration}'
    
    print(f"MIDI词表大小: {len(vocab)}")
    print(f"  特殊标记: {len(special_tokens)}")
    print(f"  事件类型: {len(event_types)}")
    print(f"  音符: 128")
    print(f"  力度级别: {len(velocity_levels)}")
    print(f"  时间偏移: 128")
    print(f"  持续时间: 127")
    
    return vocab, id_to_token


def quantize_velocity(velocity):
    """量化力度到最近的级别"""
    levels = [0, 32, 48, 64, 80, 96, 112, 127]
    return min(levels, key=lambda x: abs(x - velocity))

def sequence_to_tokens(sequence, vocab, max_len=512):
    """将MIDI事件序列转换为token IDs"""
    token_ids = [vocab['<sos>']]
    
    for event in sequence:
        if isinstance(event, tuple) and len(event) == 4:
            note, velocity, time_offset, duration = event
            
            # 量化力度
            velocity_quantized = quantize_velocity(velocity)
            
            # 编码事件：<note> note_token <velocity> velocity_token <time> time_token <duration> duration_token
            # 事件类型标记
            token_ids.append(vocab['<note>'])
            # 音符
            note_token = f'note_{note}'
            if note_token in vocab:
                token_ids.append(vocab[note_token])
            else:
                token_ids.append(vocab['<unk>'])
            
            # 力度
            token_ids.append(vocab['<velocity>'])
            vel_token = f'vel_{velocity_quantized}'
            if vel_token in vocab:
                token_ids.append(vocab[vel_token])
            else:
                token_ids.append(vocab['<unk>'])
            
            # 时间偏移
            token_ids.append(vocab['<time>'])
            time_token = f'time_{time_offset}'
            if time_token in vocab:
                token_ids.append(vocab[time_token])
            else:
                token_ids.append(vocab['<unk>'])
            
            # 持续时间
            token_ids.append(vocab['<duration>'])
            dur_token = f'dur_{duration}'
            if dur_token in vocab:
                token_ids.append(vocab[dur_token])
            else:
                token_ids.append(vocab['<unk>'])
        else:
            # 兼容旧格式（只有音符）
            if isinstance(event, int) and 0 <= event <= 127:
                note_token = f'note_{event}'
                if note_token in vocab:
                    token_ids.append(vocab[note_token])
                else:
                    token_ids.append(vocab['<unk>'])
    
    token_ids.append(vocab['<eos>'])
    
    # 填充或截断
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids = token_ids + [vocab['<pad>']] * (max_len - len(token_ids))
    
    return np.array(token_ids)


def download_maestro_dataset():
    """下载MAESTRO数据集（钢琴音乐）"""
    data_dir = "maestro_dataset"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # MAESTRO数据集下载链接（需要手动下载或使用API）
    # 这里提供一个简化的下载方案
    print("MAESTRO数据集需要从以下地址下载：")
    print("https://magenta.tensorflow.org/datasets/maestro")
    print("或者使用以下命令：")
    print("wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip")
    
    return None


def load_midi_files(directory, max_files=10000):
    """从目录递归加载MIDI文件（支持子目录）"""
    midi_files = []
    
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return None
    
    # 递归查找所有MIDI文件（包括所有子目录）
    print(f"正在搜索MIDI文件: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mid', '.midi')):
                full_path = os.path.join(root, file)
                midi_files.append(full_path)
                if len(midi_files) >= max_files:
                    print(f"已达到最大文件数限制: {max_files}")
                    break
        if len(midi_files) >= max_files:
            break
    
    print(f"找到 {len(midi_files)} 个MIDI文件（包括所有子目录）")
    return midi_files


def process_midi_dataset(midi_files, max_sequences=10000):
    """处理MIDI数据集"""
    sequences = []
    
    print("正在处理MIDI文件...")
    for midi_file in tqdm(midi_files[:max_sequences]):
        sequence = midi_to_sequence(midi_file)
        if sequence and len(sequence) > 10:  # 过滤太短的序列
            sequences.append(sequence)
    
    print(f"成功处理 {len(sequences)} 个MIDI序列")
    return sequences


# ============================================================
# 7. 训练函数
# ============================================================

def cross_entropy_loss(predictions, targets, vocab):
    """交叉熵损失"""
    batch_size, seq_len, vocab_size = predictions.shape
    predictions_flat = predictions.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    probs = softmax(predictions_flat, axis=-1)
    
    loss = 0.0
    valid_count = 0
    for i in range(len(targets_flat)):
        target_id = targets_flat[i]
        if target_id != vocab['<pad>']:
            loss -= np.log(probs[i, target_id] + 1e-10)
            valid_count += 1
    
    return loss / valid_count if valid_count > 0 else 0.0


def train_step(model, token_ids, vocab, optimizer):
    """训练一步（完整的反向传播，更新所有层）"""
    batch_size, seq_len = token_ids.shape
    
    # 输入：token_ids[:-1]，目标：token_ids[1:]
    input_ids = token_ids[:, :-1]
    target_ids = token_ids[:, 1:]
    
    # 生成causal mask
    mask = model.generate_causal_mask(input_ids.shape[1])
    
    # ========== 前向传播（保存中间结果） ==========
    # Embedding
    embedded = model.embed(input_ids)  # (batch_size, seq_len, d_model)
    model._last_embedded = embedded  # 保存用于反向传播
    model._last_input_ids = input_ids  # 保存输入ID
    
    # Decoder layers（保存每层的输出）
    x = embedded
    decoder_outputs = [x]  # 保存每层的输出
    for layer in model.decoder_layers:
        x = layer.forward(x, mask)
        decoder_outputs.append(x)
    
    # 输出层
    output = np.dot(x, model.output_proj) + model.output_bias
    
    # 计算损失
    loss = cross_entropy_loss(output, target_ids, vocab)
    
    # ========== 反向传播（完整的梯度计算） ==========
    pred_probs = softmax(output, axis=-1)
    
    # 计算目标one-hot
    target_onehot = np.zeros_like(output)
    valid_count = 0
    for b in range(batch_size):
        for s in range(target_ids.shape[1]):
            target_id = int(target_ids[b, s])
            if target_id != vocab['<pad>']:
                target_onehot[b, s, target_id] = 1.0
                valid_count += 1
    
    if valid_count == 0:
        return loss
    
    # ========== 1. 输出层反向传播 ==========
    # 计算输出层误差
    error = pred_probs - target_onehot  # (batch_size, seq_len, vocab_size)
    
    # 输出层bias的梯度
    grad_bias = np.sum(error, axis=(0, 1)) / valid_count
    optimizer.update(model.output_bias, grad_bias, "output_bias")
    
    # 输出层权重的梯度: dW = x^T @ error
    # x: (batch_size, seq_len, d_model)
    # error: (batch_size, seq_len, vocab_size)
    # output_proj: (d_model, vocab_size)
    # 梯度应该是: grad_output_proj = x^T @ error
    grad_output_proj = np.dot(x.reshape(-1, x.shape[-1]).T, 
                             error.reshape(-1, error.shape[-1])) / (batch_size * x.shape[1])
    optimizer.update(model.output_proj, grad_output_proj, "output_proj")
    
    # 计算decoder输出的梯度（通过output_proj反向传播）
    grad_decoder_output = np.dot(error, model.output_proj.T)  # (batch_size, seq_len, d_model)
    
    # ========== 2. Decoder层反向传播（从后向前） ==========
    grad_x = grad_decoder_output
    
    # 反向遍历decoder层
    for layer_idx in range(len(model.decoder_layers) - 1, -1, -1):
        layer = model.decoder_layers[layer_idx]
        grad_x = layer.backward(grad_x, mask, optimizer)
    
    # ========== 3. Embedding层反向传播 ==========
    # grad_x现在是embedding输出的梯度
    # 需要更新embedding矩阵
    
    # 计算每个token的embedding梯度
    grad_embedding = np.zeros_like(model.embedding)
    
    for b in range(batch_size):
        for s in range(input_ids.shape[1]):
            token_id = int(input_ids[b, s])
            if token_id != vocab['<pad>'] and token_id < model.vocab_size:
                grad_embedding[token_id, :] += grad_x[b, s, :]
    
    # 使用Adam更新embedding（只更新出现过的token）
    unique_tokens = np.unique(input_ids)
    for token_id in unique_tokens:
        if token_id != vocab['<pad>'] and token_id < model.vocab_size:
            token_count = np.sum(input_ids == token_id)
            if token_count > 0:
                grad_embed = grad_embedding[int(token_id), :] / token_count
                optimizer.update(
                    model.embedding[int(token_id), :], 
                    grad_embed, 
                    f"embedding_{int(token_id)}"
                )
    
    return loss


def save_model(model, vocab, id_to_token, epoch, loss, model_dir="midi_checkpoints"):
    """保存模型"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    checkpoint = {
        'model': {
            'embedding': model.embedding,
            'pos_encoding': model.pos_encoding,
            'output_proj': model.output_proj,
            'output_bias': model.output_bias,
            'decoder_layers': [],
        },
        'vocab': vocab,
        'id_to_token': id_to_token,
        'epoch': epoch,
        'loss': loss,
        'd_model': model.d_model,
        'max_len': model.max_len,
        'vocab_size': model.vocab_size,
    }
    
    # 保存decoder层
    for layer in model.decoder_layers:
        layer_data = {
            'self_attn': {
                'W_q': layer.self_attn.W_q,
                'W_k': layer.self_attn.W_k,
                'W_v': layer.self_attn.W_v,
                'W_o': layer.self_attn.W_o,
            },
            'feed_forward': {
                'W1': layer.feed_forward.W1,
                'b1': layer.feed_forward.b1,
                'W2': layer.feed_forward.W2,
                'b2': layer.feed_forward.b2,
            },
            'norm1_gamma': layer.norm1_gamma,
            'norm1_beta': layer.norm1_beta,
            'norm2_gamma': layer.norm2_gamma,
            'norm2_beta': layer.norm2_beta,
        }
        checkpoint['model']['decoder_layers'].append(layer_data)
    
    model_path = os.path.join(model_dir, f"midi_transformer_epoch_{epoch}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"模型已保存: {model_path}")
    return model_path


# ============================================================
# 8. 主训练函数
# ============================================================

def main():
    """主函数：训练MIDI音乐生成模型"""
    print("=" * 60)
    print("MIDI音乐生成Transformer训练")
    print("=" * 60)
    
    # 1. 加载MIDI数据集
    # 支持多个可能的目录位置
    possible_dirs = [
        "midi_data/maestro-v3.0.0",  # MAESTRO数据集标准位置
        "midi_data/maestro",         # 其他可能的MAESTRO位置
        "midi_data",                 # 通用目录
    ]
    
    midi_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            midi_dir = dir_path
            print(f"找到MIDI目录: {midi_dir}")
            break
    
    if midi_dir is None:
        print(f"\n未找到MIDI目录，尝试的路径：")
        for dir_path in possible_dirs:
            print(f"  - {dir_path}")
        print("\n请确保MIDI文件在以下位置之一：")
        print("  - midi_data/maestro-v3.0.0/")
        print("  - midi_data/maestro/")
        print("  - midi_data/")
        print("\n或使用以下数据集：")
        print("1. MAESTRO: https://magenta.tensorflow.org/datasets/maestro")
        print("2. Lakh MIDI: https://colinraffel.com/projects/lmd/")
        return
    
    # 加载MIDI文件（递归搜索所有子目录）
    midi_files = load_midi_files(midi_dir, max_files=10000)
    if not midi_files:
        print("未找到MIDI文件！")
        return
    
    # 处理MIDI文件
    sequences = process_midi_dataset(midi_files, max_sequences=10000)
    if not sequences:
        print("未能处理任何MIDI序列！")
        return
    
    print(f"\n成功处理 {len(sequences)} 个MIDI序列")
    
    # 2. 构建词表
    vocab, id_to_token = build_midi_vocab(sequences, max_vocab_size=512)
    vocab_size = len(vocab)
    
    print(f"词表大小: {vocab_size}")
    
    # 3. 转换为训练数据
    max_len = 512
    dataset = []
    for seq in sequences:
        token_ids = sequence_to_tokens(seq, vocab, max_len)
        dataset.append(token_ids)
    
    print(f"训练样本数: {len(dataset)}")
    
    # 打印转换后的训练数据样本（详细展示）
    print("\n" + "=" * 60)
    print("训练数据转换示例（显示前10个样本）")
    print("=" * 60)
    
    num_samples_to_show = min(10, len(dataset))
    for i in range(num_samples_to_show):
        token_ids = dataset[i]
        # 获取原始序列（从sequences中）
        original_seq = sequences[i]
        
        print(f"\n样本 {i+1}:")
        print(f"  原始MIDI事件序列长度: {len(original_seq)}")
        if original_seq and isinstance(original_seq[0], tuple):
            print(f"  原始序列前5个事件: {original_seq[:5]}")
            print(f"  事件格式: (note, velocity, time_offset, duration)")
        else:
            print(f"  原始序列前30个: {original_seq[:30]}")
        print(f"  转换后token序列长度: {len(token_ids)}")
        
        # 显示token序列（去除padding）
        non_pad_tokens = []
        for tid in token_ids:
            if tid != vocab['<pad>']:
                if tid in id_to_token:
                    token = id_to_token[tid]
                    if token not in ['<sos>', '<eos>', '<unk>', '<pad>']:
                        non_pad_tokens.append(token)
        
        print(f"  转换后前30个token: {non_pad_tokens[:30]}")
        print(f"  转换后token ID序列前30个: {[int(tid) for tid in token_ids[:30]]}")
        
        # 统计信息
        if original_seq and isinstance(original_seq[0], tuple):
            notes = [e[0] for e in original_seq if isinstance(e, tuple) and len(e) >= 1]
            velocities = [e[1] for e in original_seq if isinstance(e, tuple) and len(e) >= 2]
            time_offsets = [e[2] for e in original_seq if isinstance(e, tuple) and len(e) >= 3]
            durations = [e[3] for e in original_seq if isinstance(e, tuple) and len(e) >= 4]
            
            if notes:
                print(f"  音符范围: {min(notes)} - {max(notes)}")
                print(f"  力度范围: {min(velocities)} - {max(velocities)}")
                print(f"  时间偏移范围: {min(time_offsets)} - {max(time_offsets)}")
                print(f"  持续时间范围: {min(durations)} - {max(durations)}")
                print(f"  前3个完整事件:")
                for j, event in enumerate(original_seq[:3]):
                    if isinstance(event, tuple) and len(event) == 4:
                        print(f"    事件{j+1}: 音符={event[0]}, 力度={event[1]}, 时间偏移={event[2]}, 持续时间={event[3]}")
    
    print("\n" + "=" * 60)
    print("词表映射示例（MIDI音符 -> Token ID）")
    print("=" * 60)
    print(f"特殊标记:")
    print(f"  <pad>: {vocab['<pad>']}")
    print(f"  <sos>: {vocab['<sos>']}")
    print(f"  <eos>: {vocab['<eos>']}")
    print(f"  <unk>: {vocab['<unk>']}")
    print(f"\nMIDI音符映射示例（前20个音符）:")
    for note in range(min(20, 128)):
        if note in vocab:
            print(f"  音符 {note} -> Token ID {vocab[note]}")
    print(f"\nMIDI音符映射示例（中间20个音符，60-79）:")
    for note in range(60, min(80, 128)):
        if note in vocab:
            print(f"  音符 {note} -> Token ID {vocab[note]}")
    print(f"\nMIDI音符映射示例（最后20个音符，108-127）:")
    for note in range(108, 128):
        if note in vocab:
            print(f"  音符 {note} -> Token ID {vocab[note]}")
    
    print("\n" + "=" * 60)
    
    # 4. 初始化模型
    d_model = 256
    num_heads = 8
    num_layers = 6
    d_ff = 1024
    
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len
    )
    
    print(f"\n模型参数:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  vocab_size: {vocab_size}")
    
    # 5. 初始化Adam优化器
    learning_rate = 0.001  # Adam通常使用较小的学习率
    optimizer = AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    model.optimizer = optimizer
    
    print(f"\n优化器设置:")
    print(f"  方法: Adam")
    print(f"  学习率: {learning_rate}")
    print(f"  beta1: {optimizer.beta1}, beta2: {optimizer.beta2}")
    
    # 6. 训练循环
    print("\n开始训练...")
    epochs = 50
    checkpoint_dir = "midi_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 每个epoch使用的样本数（可以设置更小以加快训练）
    samples_per_epoch = 200  # 每个epoch使用200个样本（从12增加到200）
    print(f"每个epoch使用 {samples_per_epoch} 个样本（共 {len(dataset)} 个样本）")
    
    for epoch in range(epochs):
        # 打乱数据
        random.shuffle(dataset)
        
        # 每个epoch只使用部分数据
        epoch_dataset = dataset[:samples_per_epoch]
        
        total_loss = 0.0
        
        pbar = tqdm(epoch_dataset, desc=f"Epoch {epoch+1}/{epochs}")
        
        for token_ids in pbar:
            # 添加batch维度
            token_ids = token_ids.reshape(1, -1)
            
            # 训练一步（使用Adam优化器）
            loss = train_step(model, token_ids, vocab, optimizer)
            total_loss += loss
            
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = total_loss / len(epoch_dataset) if len(epoch_dataset) > 0 else 0.0
        print(f"\nEpoch {epoch+1}/{epochs} 完成 - 平均Loss: {avg_loss:.4f} (使用了 {len(epoch_dataset)} 个样本)")
        
        # 每个epoch保存模型
        save_model(model, vocab, id_to_token, epoch + 1, avg_loss, checkpoint_dir)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

