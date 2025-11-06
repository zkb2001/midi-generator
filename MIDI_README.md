# MIDI音乐生成Transformer

从零实现的Transformer模型，用于生成MIDI音乐。

## 功能特点

- ✅ 完全手写Transformer（不使用PyTorch/TensorFlow）
- ✅ 手动实现Q、K、V计算
- ✅ 支持MIDI音乐序列生成
- ✅ 每个epoch自动保存模型
- ✅ 简化的MIDI数据处理

## 文件说明

- `midi_transformer.py` - 主训练脚本
- `generate_midi.py` - 音乐生成脚本
- `download_midi_dataset.py` - 数据集下载工具

## 安装依赖

```bash
pip install numpy tqdm mido
```

## 使用步骤

### 1. 准备MIDI数据集

#### 方法A：使用示例MIDI（快速测试）
```bash
python download_midi_dataset.py
# 选择选项1创建示例MIDI文件
```

#### 方法B：下载真实数据集

**MAESTRO数据集（推荐，钢琴音乐）：**
```bash
# 下载地址
https://magenta.tensorflow.org/datasets/maestro

# 或直接下载
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip
mkdir -p midi_data/maestro
mv maestro-v3.0.0/* midi_data/maestro/
```

**其他免费MIDI资源：**
- https://www.midiworld.com/
- https://freemidi.org/
- https://bitmidi.com/

下载后，将所有MIDI文件放入 `midi_data/` 目录。

### 2. 训练模型

```bash
python midi_transformer.py
```

训练过程：
- 自动加载 `midi_data/` 目录下的所有MIDI文件
- 处理并转换为序列
- 构建词表（128个MIDI音符 + 特殊标记）
- 训练Transformer模型
- **每个epoch自动保存模型到 `midi_checkpoints/`**

### 3. 生成音乐

```bash
# 使用默认模型（epoch 1）
python generate_midi.py

# 使用指定模型
python generate_midi.py midi_checkpoints/midi_transformer_epoch_10.pkl
```

生成的MIDI文件保存在 `generated_music/generated.mid`

## 模型架构

- **模型类型**: GPT风格（只有Decoder）
- **d_model**: 256
- **num_heads**: 8
- **num_layers**: 6
- **d_ff**: 1024
- **词表大小**: 132 (128个MIDI音符 + 4个特殊标记)
- **最大序列长度**: 512

## 数据集要求

- **格式**: MIDI文件 (.mid, .midi)
- **推荐数量**: 至少100个文件，越多越好
- **类型**: 任何类型的MIDI音乐都可以
- **处理**: 自动提取音符序列，忽略其他信息（简化处理）

## 训练参数

- **Epochs**: 50（可在代码中修改）
- **学习率**: 0.01
- **Batch size**: 1（逐个样本训练）

## 输出说明

训练过程中：
- 每个epoch显示进度条和loss
- 每个epoch结束后自动保存模型
- 模型文件命名：`midi_transformer_epoch_{epoch}.pkl`

生成音乐时：
- 使用温度采样（temperature=0.8）
- Top-K采样（top_k=50）
- 默认生成200个音符

## 注意事项

1. **数据量**: 更多MIDI文件 = 更好的生成效果
2. **训练时间**: 取决于数据集大小和epoch数
3. **内存**: 确保有足够内存加载所有MIDI文件
4. **简化处理**: 当前版本只提取音符，不处理节奏、和弦等复杂信息

## 改进方向

- [ ] 添加节奏信息
- [ ] 支持多音轨
- [ ] 添加和弦识别
- [ ] 改进MIDI编码方式
- [ ] 支持更长序列

## 故障排除

**问题**: 找不到MIDI文件
- 解决: 确保 `midi_data/` 目录存在且包含MIDI文件

**问题**: 导入mido失败
- 解决: `pip install mido`

**问题**: 训练loss不下降
- 解决: 增加训练数据量或epoch数

**问题**: 生成的音乐不连贯
- 解决: 这是正常的，需要更多训练数据和更长的训练时间

