# 数据集使用说明

## 支持的数据集

代码支持多种方式加载对话数据集，按优先级顺序：

### 1. 本地文件（最高优先级）
将对话数据集保存为以下格式之一，放在项目根目录：

**CSV格式** (`dialogue_dataset.csv`):
```csv
hello,hi there
how are you,i am fine
what is your name,my name is ai
```

**TXT格式** (`dialogue_dataset.txt`):
```
hello
hi there
how are you
i am fine
```

### 2. Cornell Movie-Dialogs Corpus（自动下载）
- **来源**: Cornell University
- **大小**: 约22万行电影对话
- **特点**: 真实电影对话，质量较高
- **词表**: 自动限制在10000以内
- **下载**: 代码会自动下载（约1.2MB）

### 3. 内置简单数据集（备用）
如果无法下载外部数据集，会使用内置的约40个对话对。

## 推荐的数据集下载链接

### 小型对话数据集（词表<10000）：

1. **Cornell Movie-Dialogs Corpus**
   - 官方链接: https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
   - 代码会自动下载

2. **DailyDialog** (手动下载)
   - HuggingFace: https://huggingface.co/datasets/daily_dialog
   - 需要手动下载并转换为CSV格式

3. **PersonaChat** (手动下载)
   - HuggingFace: https://huggingface.co/datasets/bavard/personachat_truecased
   - 适合个性化对话

## 使用方法

### 方法1: 使用自动下载的数据集
直接运行代码，会自动尝试下载Cornell Movie-Dialogs Corpus：
```bash
python transformer_from_scratch.py
```

### 方法2: 使用本地CSV文件
1. 准备CSV文件，格式：`问题,回答`
2. 保存为 `dialogue_dataset.csv`
3. 运行代码

### 方法3: 使用本地TXT文件
1. 准备TXT文件，每两行为一组（问题+回答）
2. 保存为 `dialogue_dataset.txt`
3. 运行代码

## 词表控制

代码会自动：
- 统计所有词的频率
- 只保留最常见的10000个词
- 其他词映射为 `<unk>`
- 确保词表大小不超过10000

## 数据集大小限制

- 如果数据集超过10000个对话对，会随机采样10000个
- 词表大小限制在10000以内
- 可以通过修改代码中的 `max_vocab_size` 参数调整

## 数据预处理

代码会自动进行以下预处理：
- 转换为小写
- 移除特殊字符（只保留字母、数字、空格）
- 添加 `<sos>` 和 `<eos>` 标记
- 填充到固定长度

## 示例数据集格式

### CSV格式示例：
```csv
question,answer
hello,hi there
how are you,i am fine
what is your name,my name is ai
```

### TXT格式示例：
```
hello
hi there
how are you
i am fine
what is your name
my name is ai
```

## 注意事项

1. **编码问题**: 确保文件使用UTF-8编码
2. **格式要求**: 每行一个对话，问题在前，回答在后
3. **文本清理**: 代码会自动清理文本，移除标点符号
4. **内存限制**: 大数据集会自动采样到10000个样本

## 获取更多数据集

如果需要更多数据集，可以：
1. 从HuggingFace下载: https://huggingface.co/datasets
2. 搜索 "dialogue dataset" 或 "conversation dataset"
3. 转换为CSV或TXT格式后使用


