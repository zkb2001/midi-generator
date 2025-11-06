"""
使用训练好的MIDI Transformer模型生成音乐
"""
import numpy as np
import pickle
import os
from midi_transformer import MusicTransformer, softmax
import mido
from mido import MidiFile, MidiTrack, Message


def load_model(checkpoint_path):
    """加载训练好的模型"""
    print(f"正在加载模型: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    vocab = checkpoint['vocab']
    id_to_token = checkpoint['id_to_token']
    d_model = checkpoint['d_model']
    num_heads = checkpoint.get('num_heads', 8)
    num_layers = checkpoint.get('num_layers', 6)
    d_ff = checkpoint.get('d_ff', 1024)
    vocab_size = checkpoint['vocab_size']
    max_len = checkpoint['max_len']
    
    print(f"模型信息: epoch={checkpoint['epoch']}, loss={checkpoint['loss']:.4f}")
    
    # 重建模型
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len
    )
    
    # 加载权重
    model_data = checkpoint['model']
    model.embedding = model_data['embedding']
    model.pos_encoding = model_data['pos_encoding']
    model.output_proj = model_data['output_proj']
    model.output_bias = model_data['output_bias']
    
    # 加载decoder层
    for i, layer_data in enumerate(model_data['decoder_layers']):
        layer = model.decoder_layers[i]
        layer.self_attn.W_q = layer_data['self_attn']['W_q']
        layer.self_attn.W_k = layer_data['self_attn']['W_k']
        layer.self_attn.W_v = layer_data['self_attn']['W_v']
        layer.self_attn.W_o = layer_data['self_attn']['W_o']
        layer.feed_forward.W1 = layer_data['feed_forward']['W1']
        layer.feed_forward.b1 = layer_data['feed_forward']['b1']
        layer.feed_forward.W2 = layer_data['feed_forward']['W2']
        layer.feed_forward.b2 = layer_data['feed_forward']['b2']
        layer.norm1_gamma = layer_data['norm1_gamma']
        layer.norm1_beta = layer_data['norm1_beta']
        layer.norm2_gamma = layer_data['norm2_gamma']
        layer.norm2_beta = layer_data['norm2_beta']
    
    print("模型加载完成！")
    return model, vocab, id_to_token


def generate_music(model, vocab, id_to_token, max_length=2000, temperature=0.8, top_k=50, use_format_constraint=True):
    """生成MIDI音乐序列（带格式约束）"""
    # 初始化为<sos>
    generated = [vocab['<sos>']]
    
    print(f"开始生成音乐，最大长度: {max_length}")
    if use_format_constraint:
        print("使用格式约束模式（强制遵循事件格式）")
    
    # 事件格式状态机：0=等待<note>, 1=等待note_token, 2=等待<velocity>, 3=等待vel_token,
    #                  4=等待<time>, 5=等待time_token, 6=等待<duration>, 7=等待dur_token
    format_state = 0
    event_count = 0
    
    for step in range(max_length):
        # 显示进度（每10步或每步都显示）
        if step % 10 == 0 or step < 10:
            print(f"生成进度: {step+1}/{max_length} (已生成 {len(generated)-1} 个token, {event_count} 个事件)", end='\r')
        
        # 当前序列
        token_ids = np.array([generated])
        
        # 生成mask
        mask = model.generate_causal_mask(len(generated))
        
        # 前向传播
        output = model.forward(token_ids, mask)
        next_token_logits = output[0, -1, :]
        
        # 格式约束：根据当前状态，只允许生成特定类型的token
        if use_format_constraint:
            # 创建允许的token mask
            allowed_mask = np.full_like(next_token_logits, float('-inf'))
            
            if format_state == 0:  # 等待 <note>
                if '<note>' in vocab:
                    allowed_mask[vocab['<note>']] = next_token_logits[vocab['<note>']]
            elif format_state == 1:  # 等待 note_token
                for note in range(128):
                    note_token = f'note_{note}'
                    if note_token in vocab:
                        allowed_mask[vocab[note_token]] = next_token_logits[vocab[note_token]]
            elif format_state == 2:  # 等待 <velocity>
                if '<velocity>' in vocab:
                    allowed_mask[vocab['<velocity>']] = next_token_logits[vocab['<velocity>']]
            elif format_state == 3:  # 等待 vel_token
                for vel in [0, 32, 48, 64, 80, 96, 112, 127]:
                    vel_token = f'vel_{vel}'
                    if vel_token in vocab:
                        allowed_mask[vocab[vel_token]] = next_token_logits[vocab[vel_token]]
            elif format_state == 4:  # 等待 <time>
                if '<time>' in vocab:
                    allowed_mask[vocab['<time>']] = next_token_logits[vocab['<time>']]
            elif format_state == 5:  # 等待 time_token
                for time_val in range(128):
                    time_token = f'time_{time_val}'
                    if time_token in vocab:
                        allowed_mask[vocab[time_token]] = next_token_logits[vocab[time_token]]
            elif format_state == 6:  # 等待 <duration>
                if '<duration>' in vocab:
                    allowed_mask[vocab['<duration>']] = next_token_logits[vocab['<duration>']]
            elif format_state == 7:  # 等待 dur_token
                for dur in range(1, 128):
                    dur_token = f'dur_{dur}'
                    if dur_token in vocab:
                        allowed_mask[vocab[dur_token]] = next_token_logits[vocab[dur_token]]
            
            # 如果允许的token为空，回退到原始logits
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
        
        # Softmax
        probs = softmax(next_token_logits, axis=-1)
        
        # 采样
        next_token_id = np.random.choice(len(probs), p=probs)
        
        # 添加到序列
        generated.append(int(next_token_id))
        
        # 更新格式状态
        if use_format_constraint:
            if next_token_id in id_to_token:
                token = id_to_token[next_token_id]
                if format_state == 0 and token == '<note>':
                    format_state = 1
                elif format_state == 1 and token.startswith('note_'):
                    format_state = 2
                elif format_state == 2 and token == '<velocity>':
                    format_state = 3
                elif format_state == 3 and token.startswith('vel_'):
                    format_state = 4
                elif format_state == 4 and token == '<time>':
                    format_state = 5
                elif format_state == 5 and token.startswith('time_'):
                    format_state = 6
                elif format_state == 6 and token == '<duration>':
                    format_state = 7
                elif format_state == 7 and token.startswith('dur_'):
                    format_state = 0  # 完成一个事件，回到开始
                    event_count += 1
        
        # 如果生成<eos>，停止
        if next_token_id == vocab['<eos>']:
            print(f"\n生成完成（遇到<eos>）: 共生成 {len(generated)-1} 个token, {event_count} 个事件")
            break
    
    if len(generated) >= max_length:
        print(f"\n生成完成（达到最大长度）: 共生成 {len(generated)-1} 个token, {event_count} 个事件")
    
    # 转换为事件序列（解析token序列）
    print(f"\n开始解析token序列，共 {len(generated)} 个token")
    
    # 先打印前20个token用于调试
    print("前20个生成的token:")
    for i in range(min(20, len(generated))):
        token_id = generated[i]
        if token_id in id_to_token:
            print(f"  [{i}] token_id={token_id}, token={id_to_token[token_id]}")
        else:
            print(f"  [{i}] token_id={token_id}, token=<unknown>")
    
    events = []
    i = 0
    skipped = 0
    
    while i < len(generated):
        token_id = generated[i]
        if token_id not in id_to_token:
            i += 1
            skipped += 1
            continue
            
        token = id_to_token[token_id]
        
        # 跳过特殊标记
        if token in ['<sos>', '<eos>', '<pad>', '<unk>']:
            i += 1
            continue
        
        # 查找事件：<note> note <velocity> vel <time> time <duration> dur
        if token == '<note>' and i + 1 < len(generated):
            note_token_id = generated[i + 1]
            if note_token_id in id_to_token:
                note_token = id_to_token[note_token_id]
                if note_token.startswith('note_'):
                    try:
                        note = int(note_token.split('_')[1])
                        
                        # 查找velocity（更宽松的匹配）
                        velocity = 64  # 默认
                        if i + 2 < len(generated):
                            vel_type_token = id_to_token.get(generated[i + 2], '')
                            if vel_type_token == '<velocity>' and i + 3 < len(generated):
                                vel_token = id_to_token.get(generated[i + 3], '')
                                if vel_token.startswith('vel_'):
                                    velocity = int(vel_token.split('_')[1])
                            
                            # 查找time
                            time_offset = 0
                            if i + 4 < len(generated):
                                time_type_token = id_to_token.get(generated[i + 4], '')
                                if time_type_token == '<time>' and i + 5 < len(generated):
                                    time_token = id_to_token.get(generated[i + 5], '')
                                    if time_token.startswith('time_'):
                                        time_offset = int(time_token.split('_')[1])
                                
                                # 查找duration
                                duration = 1  # 默认
                                if i + 6 < len(generated):
                                    dur_type_token = id_to_token.get(generated[i + 6], '')
                                    if dur_type_token == '<duration>' and i + 7 < len(generated):
                                        dur_token = id_to_token.get(generated[i + 7], '')
                                        if dur_token.startswith('dur_'):
                                            duration = int(dur_token.split('_')[1])
                                        
                                        # 成功解析一个完整事件
                                        events.append((note, velocity, time_offset, duration))
                                        i += 8
                                        continue
                    except (ValueError, IndexError) as e:
                        pass
        
        # 如果找不到完整的事件格式，尝试只提取音符（兼容模式）
        if token.startswith('note_'):
            try:
                note = int(token.split('_')[1])
                # 使用默认值创建事件
                events.append((note, 64, 0, 1))
                i += 1
                continue
            except (ValueError, IndexError):
                pass
        
        i += 1
        skipped += 1
    
    print(f"解析完成: 成功解析 {len(events)} 个事件，跳过 {skipped} 个token")
    
    # 如果解析失败，至少生成一些默认事件
    if len(events) == 0:
        print("警告: 无法解析任何事件，尝试从token中提取音符...")
        for token_id in generated[:50]:  # 只检查前50个
            if token_id in id_to_token:
                token = id_to_token[token_id]
                if token.startswith('note_'):
                    try:
                        note = int(token.split('_')[1])
                        events.append((note, 64, 0, 1))
                    except (ValueError, IndexError):
                        pass
    
    return events


def events_to_midi(events, output_path, tempo=120, time_quantization=120):
    """将事件序列转换为MIDI文件（包含音符、力度、时间、持续时间）
    
    正确处理和弦：当time_offset=0时，音符同时开始
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # 设置tempo
    track.append(Message('program_change', channel=0, program=0, time=0))
    
    ticks_per_beat = mid.ticks_per_beat
    # 将量化单位转换为ticks
    # time_quantization 是毫秒，需要转换为ticks
    # 1 beat = 60000/tempo ms，所以 1ms = ticks_per_beat * tempo / 60000 ticks
    ms_to_ticks = ticks_per_beat * tempo / 60000.0
    quantization_ticks = time_quantization * ms_to_ticks
    
    # 收集所有MIDI消息（note_on和note_off），按时间排序
    midi_messages = []  # [(absolute_time, 'on'/'off', note, velocity)]
    
    last_absolute_time = 0  # 上一个事件的绝对时间（以ticks为单位）
    chord_count = 0  # 统计和弦数量
    
    for event in events:
        if isinstance(event, tuple) and len(event) == 4:
            note, velocity, time_offset, duration = event
            
            # 计算时间偏移和持续时间（转换为ticks）
            time_ticks = time_offset * quantization_ticks
            duration_ticks = duration * quantization_ticks
            
            # 计算当前事件的绝对开始时间
            current_absolute_time = last_absolute_time + time_ticks
            
            # 如果time_offset=0，说明这个音符和上一个音符同时开始（形成和弦）
            if time_offset == 0 and last_absolute_time > 0:
                chord_count += 1
            
            # 添加note_on消息
            midi_messages.append((current_absolute_time, 'on', note, velocity))
            
            # 添加note_off消息（在持续时间后）
            note_off_time = current_absolute_time + duration_ticks
            midi_messages.append((note_off_time, 'off', note, 0))
            
            # 更新last_absolute_time（用于下一个事件的时间计算）
            last_absolute_time = current_absolute_time
            
        else:
            # 兼容旧格式（只有音符）
            if isinstance(event, int) and 0 <= event <= 127:
                midi_messages.append((last_absolute_time, 'on', event, 64))
                midi_messages.append((last_absolute_time + ticks_per_beat, 'off', event, 0))
                last_absolute_time += ticks_per_beat
    
    # 按时间排序所有消息
    midi_messages.sort(key=lambda x: (x[0], 0 if x[1] == 'on' else 1))  # on在off之前（如果时间相同）
    
    # 按顺序添加MIDI消息（计算相对时间）
    last_message_time = 0
    for abs_time, msg_type, note, vel in midi_messages:
        relative_time = int(abs_time - last_message_time)
        if relative_time < 0:
            relative_time = 0  # 确保不为负
        
        if msg_type == 'on':
            track.append(Message('note_on', note=note, velocity=vel, time=relative_time))
        else:
            track.append(Message('note_off', note=note, velocity=0, time=relative_time))
        
        last_message_time = abs_time
    
    mid.save(output_path)
    print(f"MIDI文件已保存: {output_path}")
    print(f"  生成了 {len(events)} 个事件，其中 {chord_count} 个和弦事件（time_offset=0）")


def main():
    """主函数"""
    import sys
    
    # 默认模型路径
    checkpoint_path = "midi_checkpoints/midi_transformer_epoch_1.pkl"
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    try:
        # 加载模型
        model, vocab, id_to_token = load_model(checkpoint_path)
        
        # 生成音乐
        print("\n正在生成音乐...")
        events = generate_music(model, vocab, id_to_token, max_length=2000, temperature=0.8)
        
        print(f"生成了 {len(events)} 个事件")
        
        # 保存为MIDI文件
        output_dir = "generated_music"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, "generated.mid")
        events_to_midi(events, output_path)
        
        print(f"\n音乐生成完成！")
        print(f"输出文件: {output_path}")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n提示: 请先运行 midi_transformer.py 训练模型")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

