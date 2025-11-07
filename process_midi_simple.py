"""
简化版MIDI处理：将MIDI文件转换为二元组列表，并训练Transformer模型
每个音符是一个二元组：(音高, 持续时间)
- 音高: 0-127 (MIDI音符)
- 持续时间: 数字（可以是小数，单位：秒）
"""
import os
import mido
from mido import MidiFile, MidiTrack, Message
import pickle
from tqdm import tqdm
import numpy as np
import math
import random


def midi_to_events(midi_file_path):
    """将MIDI文件转换为事件列表（二元组格式）
    
    返回: [(音高, 持续时间), ...]
    """
    try:
        mid = MidiFile(midi_file_path)
        ticks_per_beat = mid.ticks_per_beat
        
        # 存储所有事件
        events = []
        active_notes = {}  # {note: start_time_ticks}
        
        # 遍历所有track
        for track in mid.tracks:
            track_time_ticks = 0
            
            for msg in track:
                track_time_ticks += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    note = msg.note
                    if 0 <= note <= 127:
                        # 记录note_on事件
                        active_notes[note] = track_time_ticks
                
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    note = msg.note
                    if note in active_notes:
                        start_time_ticks = active_notes[note]
                        duration_ticks = track_time_ticks - start_time_ticks
                        
                        # 转换为秒（使用默认tempo 120 BPM）
                        # 1 beat = 0.5秒 (120 BPM)
                        ticks_to_seconds = lambda t: t / ticks_per_beat * 0.5
                        
                        duration_seconds = ticks_to_seconds(duration_ticks)
                        
                        # 确保持续时间大于0
                        if duration_seconds <= 0:
                            duration_seconds = 0.1  # 最小0.1秒
                        
                        # 添加事件：(音高, 持续时间)
                        events.append((
                            int(note),           # 音高 (0-127)
                            float(duration_seconds)  # 持续时间（秒，可以是小数）
                        ))
                        
                        del active_notes[note]
        
        # 过滤太短的序列
        if len(events) < 5:
            return None
        
        return events
    
    except Exception as e:
        print(f"处理MIDI文件失败 {midi_file_path}: {e}")
        return None


def load_midi_files(directory):
    """从目录加载所有MIDI文件"""
    midi_files = []
    
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return []
    
    print(f"正在搜索MIDI文件: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mid', '.midi')):
                full_path = os.path.join(root, file)
                midi_files.append(full_path)
    
    print(f"找到 {len(midi_files)} 个MIDI文件")
    return midi_files


def process_dataset(midi_files):
    """处理MIDI数据集，返回所有事件列表"""
    all_sequences = []
    failed_count = 0
    
    print("正在处理MIDI文件...")
    for midi_file in tqdm(midi_files):
        events = midi_to_events(midi_file)
        if events is not None:
            all_sequences.append(events)
        else:
            failed_count += 1
    
    print(f"\n处理统计:")
    print(f"  总文件数: {len(midi_files)}")
    print(f"  成功处理: {len(all_sequences)} 个序列")
    print(f"  处理失败: {failed_count} 个文件")
    
    return all_sequences


def print_sample_sequences(sequences, num_samples=3):
    """打印几个样本序列的详细信息"""
    print("\n" + "=" * 80)
    print(f"样本序列展示（前 {num_samples} 个）")
    print("=" * 80)
    
    for i in range(min(num_samples, len(sequences))):
        seq = sequences[i]
        print(f"\n样本 {i+1}:")
        print(f"  序列长度: {len(seq)} 个音符")
        print(f"  前10个音符（二元组格式）:")
        for j, event in enumerate(seq[:10]):
            if len(event) == 2:
                note, duration = event
                print(f"    音符 {j+1}: (音高={note:3d}, 持续时间={duration:6.3f}秒)")
            elif len(event) == 4:
                note, duration, relative_time, velocity = event
                print(f"    音符 {j+1}: (音高={note:3d}, 持续时间={duration:6.3f}秒, 相对时间={relative_time:6.3f}秒, 力度={velocity:3d})")
        
        # 统计信息
        notes = [e[0] for e in seq]
        durations = [e[1] for e in seq]
        
        print(f"\n  统计信息:")
        print(f"    音高范围: {min(notes)} - {max(notes)}")
        print(f"    持续时间范围: {min(durations):.3f} - {max(durations):.3f} 秒")
        
        # 计算总时长（所有音符持续时间之和）
        total_time = sum(durations) if durations else 0
        print(f"    总时长: {total_time:.2f} 秒")


def save_sequences(sequences, output_path="midi_sequences.pkl"):
    """保存处理好的序列"""
    with open(output_path, 'wb') as f:
        pickle.dump(sequences, f)
    print(f"\n序列已保存到: {output_path}")
    print(f"  共 {len(sequences)} 个序列")


def main():
    """主函数"""
    print("=" * 80)
    print("MIDI文件处理 - 二元组格式（音高, 持续时间）")
    print("=" * 80)
    
    # MIDI文件目录
    midi_dir = r"D:\um study\DL\big_project\nottingham-dataset-master\MIDI\melody"
    
    if not os.path.exists(midi_dir):
        print(f"错误: 目录不存在: {midi_dir}")
        print("请检查路径是否正确")
        return
    
    # 1. 加载MIDI文件
    midi_files = load_midi_files(midi_dir)
    if not midi_files:
        print("未找到MIDI文件！")
        return
    
    # 2. 处理MIDI文件
    sequences = process_dataset(midi_files)
    if not sequences:
        print("未能处理任何MIDI序列！")
        return
    
    # 3. 打印样本
    print_sample_sequences(sequences, num_samples=5)
    
    # 4. 保存序列
    save_sequences(sequences, "midi_sequences.pkl")
    
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)


# ============================================================
# Transformer模型组件（从零实现）
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
    # 确保dout和x的形状匹配
    if dout.shape != x.shape:
        # 如果形状不匹配，尝试调整
        min_seq_len = min(dout.shape[1], x.shape[1])
        dout = dout[:, :min_seq_len, :]
        x = x[:, :min_seq_len, :]
    
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    normalized = (x - mean) / std
    
    dgamma = np.sum(dout * normalized, axis=(0, 1))
    dbeta = np.sum(dout, axis=(0, 1))
    
    dnormalized = dout * gamma
    dvar = np.sum(dnormalized * (x - mean) * -0.5 / (std ** 3), axis=-1, keepdims=True)
    dmean = np.sum(dnormalized * -1.0 / std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)
    dx = dnormalized / std + dvar * 2.0 * (x - mean) / x.shape[-1] + dmean / x.shape[-1]
    
    return dx, dgamma, dbeta


def positional_encoding(seq_len, d_model):
    """位置编码（sinusoidal）"""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


class MultiHeadAttention:
    """多头注意力机制"""
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_q, _ = query.shape
        Q = np.dot(query, self.W_q)
        K = np.dot(key, self.W_k)
        V = np.dot(value, self.W_v)
        
        Q = Q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, key.shape[1], self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, value.shape[1], self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])
            scores = np.where(mask, -1e9, scores)
        
        attention_weights = softmax(scores, axis=-1)
        attention_output = np.matmul(attention_weights, V)
        self.attention_output_multihead = attention_output  # 保存多头格式 (batch_size, num_heads, seq_len, head_dim)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)
        output = np.dot(attention_output, self.W_o)
        
        # 保存用于反向传播的中间结果（多头格式）
        self.query = query
        self.key = key
        self.value = value
        self.Q = Q  # 保存多头格式的Q
        self.K = K  # 保存多头格式的K
        self.V = V  # 保存多头格式的V
        self.attention_output = attention_output  # 保存合并后的格式 (batch_size, seq_len, d_model)
        self.attention_weights = attention_weights
        self.mask = mask
        return output, attention_weights
    
    def backward(self, dout):
        batch_size, seq_len_q, d_model = dout.shape
        
        # 步骤8反向: 输出投影
        dattention_output = np.dot(dout, self.W_o.T)  # (batch_size, seq_len_q, d_model)
        
        # 步骤7反向: 合并多头
        dattention_output_reshaped = dattention_output.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim)
        dattention_output_transposed = dattention_output_reshaped.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len_q, head_dim)
        
        # 步骤6反向: 加权求和
        # dV = attention_weights^T @ dattention_output
        dV = np.matmul(self.attention_weights.transpose(0, 1, 3, 2), dattention_output_transposed)  # (batch_size, num_heads, seq_len_v, head_dim)
        # dattention_weights = dattention_output @ V^T
        dattention_weights = np.matmul(dattention_output_transposed, self.V.transpose(0, 1, 3, 2))  # (batch_size, num_heads, seq_len_q, seq_len_v)
        
        # 步骤5反向: Softmax（简化处理）
        dscores = dattention_weights / math.sqrt(self.head_dim)
        
        dQ = np.matmul(dscores, self.K)  # (batch_size, num_heads, seq_len_q, head_dim)
        dK = np.matmul(dscores.transpose(0, 1, 3, 2), self.Q)  # (batch_size, num_heads, seq_len_k, head_dim)
        
        dQ = dQ.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, d_model)
        dK = dK.transpose(0, 2, 1, 3).reshape(batch_size, self.key.shape[1], d_model)
        dV = dV.transpose(0, 2, 1, 3).reshape(batch_size, self.value.shape[1], d_model)
        
        dquery = np.dot(dQ, self.W_q.T)
        dkey = np.dot(dK, self.W_k.T)
        dvalue = np.dot(dV, self.W_v.T)
        
        dW_q = np.dot(self.query.reshape(-1, d_model).T, dQ.reshape(-1, d_model)) / batch_size
        dW_k = np.dot(self.key.reshape(-1, d_model).T, dK.reshape(-1, d_model)) / batch_size
        dW_v = np.dot(self.value.reshape(-1, d_model).T, dV.reshape(-1, d_model)) / batch_size
        dW_o = np.dot(self.attention_output.reshape(-1, d_model).T, dout.reshape(-1, d_model)) / batch_size
        
        return dquery, dkey, dvalue, dW_q, dW_k, dW_v, dW_o


class FeedForward:
    """前馈神经网络"""
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        self.input = x
        x = np.dot(x, self.W1) + self.b1
        self.pre_activation = x
        x = np.maximum(0, x)
        self.activation = x
        x = np.dot(x, self.W2) + self.b2
        return x
    
    def backward(self, dout):
        batch_size, seq_len, d_model = dout.shape
        dactivation = np.dot(dout, self.W2.T)
        dW2 = np.dot(self.activation.reshape(-1, self.activation.shape[-1]).T, 
                    dout.reshape(-1, dout.shape[-1])) / (batch_size * seq_len)
        db2 = np.sum(dout, axis=(0, 1))
        dpre_activation = dactivation * (self.pre_activation > 0)
        dx = np.dot(dpre_activation, self.W1.T)
        dW1 = np.dot(self.input.reshape(-1, self.input.shape[-1]).T, 
                    dpre_activation.reshape(-1, dpre_activation.shape[-1])) / (batch_size * seq_len)
        db1 = np.sum(dpre_activation, axis=(0, 1))
        return dx, dW1, db1, dW2, db2


class TransformerDecoderLayer:
    """Transformer解码器层"""
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1_gamma = np.ones(d_model)
        self.norm1_beta = np.zeros(d_model)
        self.norm2_gamma = np.ones(d_model)
        self.norm2_beta = np.zeros(d_model)
    
    def forward(self, x, mask=None):
        self.input = x
        attn_output, _ = self.self_attn.forward(x, x, x, mask)
        self.attn_output = attn_output
        self.pre_norm1 = x + attn_output
        x = layer_norm(self.pre_norm1, self.norm1_gamma, self.norm1_beta)
        self.post_norm1 = x
        ff_output = self.feed_forward.forward(x)
        self.ff_output = ff_output
        self.pre_norm2 = x + ff_output
        x = layer_norm(self.pre_norm2, self.norm2_gamma, self.norm2_beta)
        self.post_norm2 = x
        return x
    
    def backward(self, dout, mask=None, optimizer=None):
        # 确保dout的形状与保存的中间结果匹配
        if hasattr(self, 'pre_norm2') and dout.shape != self.pre_norm2.shape:
            # 如果形状不匹配，截断或填充
            min_seq_len = min(dout.shape[1], self.pre_norm2.shape[1])
            dout = dout[:, :min_seq_len, :]
            if self.pre_norm2.shape[1] > min_seq_len:
                self.pre_norm2 = self.pre_norm2[:, :min_seq_len, :]
            if self.pre_norm1.shape[1] > min_seq_len:
                self.pre_norm1 = self.pre_norm1[:, :min_seq_len, :]
            if self.attn_output.shape[1] > min_seq_len:
                self.attn_output = self.attn_output[:, :min_seq_len, :]
            if self.input.shape[1] > min_seq_len:
                self.input = self.input[:, :min_seq_len, :]
            if hasattr(self, 'post_norm1') and self.post_norm1.shape[1] > min_seq_len:
                self.post_norm1 = self.post_norm1[:, :min_seq_len, :]
            if hasattr(self, 'ff_output') and self.ff_output.shape[1] > min_seq_len:
                self.ff_output = self.ff_output[:, :min_seq_len, :]
        
        dx_norm2, dgamma2, dbeta2 = layer_norm_backward(dout, self.pre_norm2, self.norm2_gamma, self.norm2_beta)
        dff_output = dx_norm2
        dx_ff, dW1, db1, dW2, db2 = self.feed_forward.backward(dff_output)
        dx_norm1 = dx_ff + dx_norm2
        dx_pre_norm1, dgamma1, dbeta1 = layer_norm_backward(dx_norm1, self.pre_norm1, self.norm1_gamma, self.norm1_beta)
        dattn_output = dx_pre_norm1
        dx_attn, _, _, dW_q, dW_k, dW_v, dW_o = self.self_attn.backward(dattn_output)
        dx = dx_attn + dx_attn + dx_attn
        
        if optimizer is not None:
            optimizer.update(self.self_attn.W_q, dW_q, "layer_attn_W_q")
            optimizer.update(self.self_attn.W_k, dW_k, "layer_attn_W_k")
            optimizer.update(self.self_attn.W_v, dW_v, "layer_attn_W_v")
            optimizer.update(self.self_attn.W_o, dW_o, "layer_attn_W_o")
            optimizer.update(self.feed_forward.W1, dW1, "layer_ff_W1")
            optimizer.update(self.feed_forward.b1, db1, "layer_ff_b1")
            optimizer.update(self.feed_forward.W2, dW2, "layer_ff_W2")
            optimizer.update(self.feed_forward.b2, db2, "layer_ff_b2")
            optimizer.update(self.norm1_gamma, dgamma1, "layer_norm1_gamma")
            optimizer.update(self.norm1_beta, dbeta1, "layer_norm1_beta")
            optimizer.update(self.norm2_gamma, dgamma2, "layer_norm2_gamma")
            optimizer.update(self.norm2_beta, dbeta2, "layer_norm2_beta")
        
        return dx


class AdamOptimizer:
    """Adam优化器"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}
    
    def get_state_key(self, param_name, param_id):
        return f"{param_name}_{id(param_id)}"
    
    def update(self, param, grad, param_name="param"):
        self.t += 1
        state_key = self.get_state_key(param_name, param)
        if state_key not in self.m:
            self.m[state_key] = np.zeros_like(param)
            self.v[state_key] = np.zeros_like(param)
        m = self.m[state_key]
        v = self.v[state_key]
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.m[state_key] = m
        self.v[state_key] = v


class MusicTransformer:
    """音乐生成Transformer模型"""
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, d_ff=1024, max_len=5120):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_encoding = positional_encoding(max_len, d_model)
        self.decoder_layers = [TransformerDecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02
        self.output_bias = np.zeros(vocab_size)
        self.optimizer = None
    
    def embed(self, token_ids):
        batch_size, seq_len = token_ids.shape
        embedded = self.embedding[token_ids]
        pos_enc = self.pos_encoding[:seq_len]
        return embedded + pos_enc
    
    def forward(self, token_ids, mask=None):
        x = self.embed(token_ids)
        for layer in self.decoder_layers:
            x = layer.forward(x, mask)
        output = np.dot(x, self.output_proj) + self.output_bias
        return output
    
    def generate_causal_mask(self, seq_len):
        return np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)


# ============================================================
# 词表构建和token化（适配四元组格式）
# ============================================================

def quantize_time(time_seconds, time_resolution=0.05):
    """量化时间（秒）到整数索引
    例如：0.05秒分辨率，0.0-0.05 -> 0, 0.05-0.10 -> 1, ...
    """
    return int(time_seconds / time_resolution)


def dequantize_time(time_index, time_resolution=0.05):
    """反量化时间索引到秒"""
    return time_index * time_resolution


def build_vocab(sequences, time_resolution=0.05, max_duration=5.0):
    """构建词表（2-token方案：note, duration）"""
    vocab = {}
    id_to_token = {}
    
    # 特殊标记
    #<pad>means padding, <sos>means start of sequence, <eos>means end of sequence, <unk>means unknown token
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    for i, token in enumerate(special_tokens):
        vocab[token] = i
        id_to_token[i] = token
    
    # MIDI音符 (0-127)
    for note in range(128):
        idx = len(vocab)
        vocab[f'note_{note}'] = idx
        id_to_token[idx] = f'note_{note}'
    
    # 持续时间（量化，0.05秒分辨率，最多5秒 = 100个级别）
    max_duration_idx = int(max_duration / time_resolution)
    for dur_idx in range(max_duration_idx + 1):
        idx = len(vocab)
        vocab[f'dur_{dur_idx}'] = idx
        id_to_token[idx] = f'dur_{dur_idx}'
    
    print(f"词表大小: {len(vocab)}")
    print(f"  特殊标记: {len(special_tokens)}")
    print(f"  音符: 128")
    print(f"  持续时间级别: {max_duration_idx + 1}")
    print(f"  每个事件: 2个token (note, duration)")
    
    return vocab, id_to_token, time_resolution


def sequence_to_tokens(sequence, vocab, max_len=5120, time_resolution=0.05):
    """将二元组序列转换为token IDs（2-token方案：note, duration）"""
    token_ids = [vocab['<sos>']]
    
    for event in sequence:
        if isinstance(event, tuple):
            # 兼容旧格式（4元组）和新格式（2元组）
            if len(event) == 2:
                note, duration = event
            elif len(event) == 4:
                note, duration, _, _ = event  # 忽略相对时间和力度
            else:
                continue
            
            # 量化持续时间
            duration_idx = min(quantize_time(duration, time_resolution), 100)
            
            # 添加2个token
            token_ids.append(vocab.get(f'note_{note}', vocab['<unk>']))
            token_ids.append(vocab.get(f'dur_{duration_idx}', vocab['<unk>']))
    
    token_ids.append(vocab['<eos>'])
    
    # 填充或截断
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids = token_ids + [vocab['<pad>']] * (max_len - len(token_ids))
    
    return np.array(token_ids)


# ============================================================
# 训练函数
# ============================================================

def train_step(model, batch, vocab, optimizer, time_resolution=0.05):
    """训练一步"""
    batch_size = len(batch)
    max_seq_len = max(len(seq) for seq in batch)
    
    # 转换为token序列
    token_sequences = []
    for seq in batch:
        tokens = sequence_to_tokens(seq, vocab, max_len=max_seq_len + 10, time_resolution=time_resolution)
        token_sequences.append(tokens)
    
    # 找到实际最大长度（去除padding）
    actual_max_len = 0
    for tokens in token_sequences:
        # 找到最后一个非padding的位置
        non_pad_len = len(tokens)
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] != vocab['<pad>']:
                non_pad_len = i + 1
                break
        actual_max_len = max(actual_max_len, non_pad_len)
    
    # 截断到实际最大长度
    token_sequences = [tokens[:actual_max_len] for tokens in token_sequences]
    
    # 填充到相同长度
    padded_sequences = []
    for tokens in token_sequences:
        if len(tokens) < actual_max_len:
            tokens = np.concatenate([tokens, [vocab['<pad>']] * (actual_max_len - len(tokens))])
        padded_sequences.append(tokens)
    
    token_ids = np.array(padded_sequences)
    
    # 创建mask
    mask = model.generate_causal_mask(actual_max_len)
    
    # 前向传播（使用完整的token_ids）
    output = model.forward(token_ids, mask)
    
    # 计算损失（交叉熵）：预测下一个token
    # targets是token_ids向右移动一位（移除了<sos>）
    targets = token_ids[:, 1:]  # (batch_size, seq_len-1)
    logits = output[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)
    
    # 计算损失
    batch_size, seq_len, vocab_size = logits.shape
    loss = 0.0
    correct = 0
    total = 0
    
    for b in range(batch_size):
        for s in range(seq_len):
            if targets[b, s] != vocab['<pad>']:
                probs = softmax(logits[b, s, :], axis=-1)
                loss -= np.log(probs[targets[b, s]] + 1e-10)
                if np.argmax(probs) == targets[b, s]:
                    correct += 1
                total += 1
    
    loss = loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    # 反向传播
    error = np.zeros_like(logits)
    for b in range(batch_size):
        for s in range(seq_len):
            if targets[b, s] != vocab['<pad>']:
                probs = softmax(logits[b, s, :], axis=-1)
                error[b, s, :] = probs
                error[b, s, targets[b, s]] -= 1.0
    
    error = error / total if total > 0 else error
    
    # 反向传播到模型
    # 注意：x需要与logits对齐（不包括最后一个位置）
    x = model.embed(token_ids[:, :-1])  # (batch_size, seq_len-1, d_model)
    
    # 从输出层开始反向传播
    dx = np.dot(error, model.output_proj.T)  # (batch_size, seq_len-1, d_model)
    
    # 更新输出层
    grad_output_proj = np.dot(x.reshape(-1, x.shape[-1]).T, error.reshape(-1, error.shape[-1])) / (batch_size * seq_len)
    grad_output_bias = np.sum(error, axis=(0, 1)) / (batch_size * seq_len)
    
    optimizer.update(model.output_proj, grad_output_proj, "output_proj")
    optimizer.update(model.output_bias, grad_output_bias, "output_bias")
    
    # 反向传播通过decoder layers
    # 需要重新前向传播以获取正确的中间结果（只到seq_len）
    # 因为前向传播时使用的是完整序列(actual_max_len)，但反向传播时只需要seq_len
    token_ids_input = token_ids[:, :-1]  # 移除最后一个token，用于反向传播
    mask_backward = model.generate_causal_mask(seq_len)
    x_input = model.embed(token_ids_input)
    
    # 重新前向传播以获取正确的中间结果（只到seq_len）
    for layer in model.decoder_layers:
        x_input = layer.forward(x_input, mask_backward)
    
    # 现在反向传播（dx的形状是(batch_size, seq_len, d_model)）
    for layer_idx in range(len(model.decoder_layers) - 1, -1, -1):
        layer = model.decoder_layers[layer_idx]
        dx = layer.backward(dx, mask_backward, optimizer)
    
    # 更新embedding（从dx中提取）
    grad_embedding = np.zeros_like(model.embedding)
    for b in range(batch_size):
        for s in range(seq_len):
            if targets[b, s] != vocab['<pad>']:
                token_id = token_ids[b, s]
                grad_embedding[token_id] += dx[b, s, :]
    
    # 更新embedding参数
    for token_id in range(model.vocab_size):
        if np.any(grad_embedding[token_id] != 0):
            grad = grad_embedding[token_id:token_id+1].reshape(1, -1)
            optimizer.update(model.embedding[token_id:token_id+1], grad, f"embedding_{token_id}")
    
    return loss, accuracy


def train_model(sequences, vocab, id_to_token, time_resolution, epochs=10, batch_size=4, 
                samples_per_epoch=50, d_model=256, num_heads=8, num_layers=4, d_ff=1024):
    """训练模型"""
    print(f"\n开始训练模型...")
    print(f"  训练样本数: {len(sequences)}")
    print(f"  每个epoch样本数: {samples_per_epoch}")
    print(f"  总epoch数: {epochs}")
    print(f"  Batch大小: {batch_size}")
    
    # 初始化模型
    vocab_size = len(vocab)
    model = MusicTransformer(vocab_size, d_model=d_model, num_heads=num_heads, 
                             num_layers=num_layers, d_ff=d_ff, max_len=5120)
    optimizer = AdamOptimizer(learning_rate=0.0001)
    model.optimizer = optimizer
    
    # 训练循环
    checkpoint_dir = "midi_checkpoints_simple"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        # 随机选择样本
        selected_sequences = random.sample(sequences, min(samples_per_epoch, len(sequences)))
        
        for i in tqdm(range(0, len(selected_sequences), batch_size)):
            batch = selected_sequences[i:i+batch_size]
            loss, accuracy = train_step(model, batch, vocab, optimizer, time_resolution)
            epoch_loss += loss
            epoch_accuracy += accuracy
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0.0
        
        print(f"  平均损失: {avg_loss:.4f}, 平均准确率: {avg_accuracy:.4f}")
        
        # 保存checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'vocab': vocab,
                'id_to_token': id_to_token,
                'time_resolution': time_resolution,
                'epoch': epoch + 1
            }, f)
        print(f"  已保存checkpoint: {checkpoint_path}")
    
    return model


# ============================================================
# 生成函数
# ============================================================
#top_k指的是从下一个token的logits中选择前k个最大的token，logits是每个token的logits值
#选择前k个是为了防止生成过于相似的token，导致生成的音乐过于单调
def generate_music(model, vocab, id_to_token, time_resolution, max_events=1000, 
                  temperature=0.8, top_k=50):
    """生成音乐序列（2-token方案：note, duration）"""
    generated = [vocab['<sos>']]
    format_state = 0  # 0=note, 1=duration
    event_count = 0
    
    print(f"开始生成音乐（最多 {max_events} 个事件）...")
    
    for step in range(max_events * 2 + 10):  # 每个事件2个token
        token_ids = np.array([generated])
        mask = model.generate_causal_mask(len(generated))
        output = model.forward(token_ids, mask)
        next_token_logits = output[0, -1, :]
        
        # 格式约束
        allowed_mask = np.full_like(next_token_logits, float('-inf'))
        if format_state == 0:  # 等待note
            for note in range(128):
                token = f'note_{note}'
                if token in vocab:
                    allowed_mask[vocab[token]] = next_token_logits[vocab[token]]
        elif format_state == 1:  # 等待duration
            for dur_idx in range(101):
                token = f'dur_{dur_idx}'
                if token in vocab:
                    allowed_mask[vocab[token]] = next_token_logits[vocab[token]]
        
        if np.max(allowed_mask) == float('-inf'):
            allowed_mask = next_token_logits.copy()
        else:
            next_token_logits = allowed_mask
        
        # Top-K采样
        if top_k > 0:
            top_k_indices = np.argsort(next_token_logits)[-top_k:]
            top_k_logits = next_token_logits[top_k_indices]
            mask_logits = np.full_like(next_token_logits, float('-inf'))
            mask_logits[top_k_indices] = top_k_logits
            next_token_logits = mask_logits
        
        # 温度采样
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        probs = softmax(next_token_logits, axis=-1)
        next_token_id = np.random.choice(len(probs), p=probs)
        generated.append(int(next_token_id))
        
        # 更新格式状态
        if next_token_id in id_to_token:
            token = id_to_token[next_token_id]
            if format_state == 0 and token.startswith('note_'):
                format_state = 1
            elif format_state == 1 and token.startswith('dur_'):
                format_state = 0
                event_count += 1
                if event_count >= max_events:
                    break
        
        if next_token_id == vocab['<eos>']:
            break
    
    # 解析生成的事件（2-token方案）
    events = []
    i = 0
    while i < len(generated) - 1:
        if generated[i] in id_to_token:
            token = id_to_token[generated[i]]
            if token.startswith('note_') and i + 1 < len(generated):
                try:
                    note = int(token.split('_')[1])
                    dur_token = id_to_token.get(generated[i+1], '')
                    
                    if dur_token.startswith('dur_'):
                        dur_idx = int(dur_token.split('_')[1])
                        duration = dequantize_time(dur_idx, time_resolution)
                    else:
                        duration = 0.5
                    
                    events.append((note, duration))
                    i += 2
                    continue
                except (ValueError, IndexError):
                    pass
        i += 1
    
    return events


def events_to_midi(events, output_path="generated_simple.mid", tempo=120):
    """将事件列表转换为MIDI文件（2-token方案：note, duration）"""
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # 兼容旧格式（4元组）和新格式（2元组）
    for event in events:
        if len(event) == 2:
            note, duration = event
            velocity = 64  # 默认力度
            relative_time = 0.0  # 默认无延迟
        elif len(event) == 4:
            note, duration, relative_time, velocity = event
        else:
            continue
        
        # 添加相对时间延迟
        if relative_time > 0:
            ticks_delay = int(relative_time * mid.ticks_per_beat * 2)  # 假设120 BPM
            track.append(Message('note_on', note=note, velocity=velocity, time=ticks_delay))
        else:
            track.append(Message('note_on', note=note, velocity=velocity, time=0))
        
        # 添加持续时间
        ticks_duration = int(duration * mid.ticks_per_beat * 2)
        track.append(Message('note_off', note=note, velocity=0, time=ticks_duration))
    
    mid.save(output_path)
    print(f"MIDI文件已保存到: {output_path}")


# ============================================================
# 主函数（训练和生成）
# ============================================================

def main_train_and_generate():
    """主函数：训练模型并生成音乐"""
    print("=" * 80)
    print("MIDI Transformer训练和生成")
    print("=" * 80)
    
    # MIDI文件目录
    midi_dir = r"D:\um study\DL\big_project\nottingham-dataset-master\MIDI\melody"
    
    if not os.path.exists(midi_dir):
        print(f"错误: 目录不存在: {midi_dir}")
        return
    
    # 1. 加载和处理MIDI文件
    midi_files = load_midi_files(midi_dir)
    if not midi_files:
        print("未找到MIDI文件！")
        return
    
    sequences = process_dataset(midi_files)
    if not sequences:
        print("未能处理任何MIDI序列！")
        return
    
    # 2. 构建词表
    vocab, id_to_token, time_resolution = build_vocab(sequences)
    
    # 3. 训练模型
    model = train_model(sequences, vocab, id_to_token, time_resolution, 
                       epochs=100, batch_size=4, samples_per_epoch=50)
    
    # 4. 生成音乐
    print("\n" + "=" * 80)
    print("生成音乐...")
    print("=" * 80)
    generated_events = generate_music(model, vocab, id_to_token, time_resolution, 
                                      max_events=50, temperature=0.8, top_k=50)
    
    print(f"生成了 {len(generated_events)} 个事件")
    print("前10个事件:")
    for i, event in enumerate(generated_events[:10]):
        if len(event) == 2:
            note, duration = event
            print(f"  事件 {i+1}: 音高={note}, 持续时间={duration:.3f}秒")
        elif len(event) == 4:
            note, duration, relative_time, velocity = event
            print(f"  事件 {i+1}: 音高={note}, 持续时间={duration:.3f}秒, 相对时间={relative_time:.3f}秒, 力度={velocity}")
    
    # 5. 保存为MIDI文件
    events_to_midi(generated_events, "generated_simple.mid")
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    # 默认同时处理数据和训练
    if len(sys.argv) > 1 and sys.argv[1] == "data_only":
        main()  # 只处理数据
    else:
        main_train_and_generate()  # 处理数据并训练

