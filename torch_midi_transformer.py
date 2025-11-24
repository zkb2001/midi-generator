"""
基于PyTorch的MIDI Transformer训练脚本

请先在 PowerShell 中激活 DL_class 虚拟环境:
    `conda activate DL_class` 或 `.\DL_class\Scripts\activate`

然后运行:
    python torch_midi_transformer.py --midi_dir "<MIDI目录>"

该脚本复用了 `process_midi_simple.py` 中的数据处理逻辑，并借鉴
`minimind/trainer/train_pretrain.py` 的训练流程（混合精度 / 梯度累积 / 断点续训）。
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from process_midi_simple import (
    build_vocab,
    load_midi_files,
    process_dataset,
    sequence_to_tokens,
)


# ============================
# 实用函数
# ============================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    step: int,
    vocab: Dict[str, int],
    id_to_token: Dict[int, str],
    time_resolution: float,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "vocab": vocab,
        "id_to_token": id_to_token,
        "time_resolution": time_resolution,
    }
    torch.save(state, path)
    print(f"[Checkpoint] 已保存到 {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Tuple[int, int, Dict[str, int], Dict[int, str], float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    print(f"[Checkpoint] 已加载 {path}")
    return (
        ckpt.get("epoch", 0),
        ckpt.get("step", 0),
        ckpt["vocab"],
        ckpt["id_to_token"],
        ckpt["time_resolution"],
    )


# ============================
# 数据集
# ============================


class MidiTokenDataset(Dataset):
    """将MIDI事件转为token并提供训练样本"""

    def __init__(
        self,
        sequences: List[List[Tuple[int, float]]],
        vocab: Dict[str, int],
        max_len: int,
        time_resolution: float,
    ) -> None:
        self.vocab = vocab
        self.inputs = []
        self.targets = []
        self.attention_masks = []
        self.loss_masks = []

        for seq in sequences:
            tokens = sequence_to_tokens(
                seq,
                vocab,
                max_len=max_len,
                time_resolution=time_resolution,
            )
            # 输入去掉最后一个token，标签去掉第一个token
            input_ids = tokens[:-1]
            target_ids = tokens[1:]
            attn_mask = (input_ids != vocab["<pad>"]).astype(np.float32)
            loss_mask = (target_ids != vocab["<pad>"]).astype(np.float32)

            self.inputs.append(torch.from_numpy(input_ids).long())
            self.targets.append(torch.from_numpy(target_ids).long())
            self.attention_masks.append(torch.from_numpy(attn_mask).float())
            self.loss_masks.append(torch.from_numpy(loss_mask).float())

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[idx],
            "target_ids": self.targets[idx],
            "attention_mask": self.attention_masks[idx],
            "loss_mask": self.loss_masks[idx],
        }


# ============================
# 模型
# ============================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    """RoPE相对位置编码"""

    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.zeros(1, 1, 1, dim), persistent=False)
        self.register_buffer("sin_cached", torch.zeros(1, 1, 1, dim), persistent=False)

    def _update_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            seq_len > self.max_seq_len_cached
            or self.cos_cached.device != device
            or self.cos_cached.dtype != dtype
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(dtype=dtype)
            sin = emb.sin().to(dtype=dtype)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            self.cos_cached = cos
            self.sin_cached = sin
            self.max_seq_len_cached = seq_len

    def get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin(seq_len, device, dtype)
        return (
            self.cos_cached[:, :, :seq_len, :].to(device=device, dtype=dtype),
            self.sin_cached[:, :, :seq_len, :].to(device=device, dtype=dtype),
        )


class RoPEMultiheadAttention(nn.Module):
    """带RoPE的多头自注意力"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb.get_cos_sin(seq_len, q.device, q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        if key_padding_mask is not None:
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)
        return attn_output


class TransformerBlock(nn.Module):
    """单个Transformer层，带RoPE注意力"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiheadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output = self.attn(attn_input, attn_mask, key_padding_mask)
        x = x + attn_output

        ff_input = self.norm2(x)
        ff_output = self.ff(ff_input)
        x = x + ff_output
        return x


class MidiTransformer(nn.Module):
    """基于自定义RoPE注意力的Transformer语言模型"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.embed_dropout(x)

        seq_len = input_ids.size(1)
        device = input_ids.device
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        key_padding = (
            (attention_mask == 0).to(torch.bool) if attention_mask is not None else None
        )

        for layer in self.layers:
            x = layer(x, causal_mask, key_padding)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


# ============================
# 训练流程
# ============================


@dataclass
class TrainConfig:
    midi_dir: str
    cache_path: str
    max_len: int
    batch_size: int
    epochs: int
    lr: float
    accumulation_steps: int
    grad_clip: float
    device: str
    precision: str
    num_workers: int
    save_dir: str
    save_interval: int
    resume_path: Optional[str]
    seed: int


def prepare_sequences(cfg: TrainConfig) -> Tuple[List, Dict, Dict, float]:
    if cfg.cache_path and os.path.exists(cfg.cache_path):
        with open(cfg.cache_path, "rb") as f:
            sequences = pickle.load(f)
        print(f"[Data] 已从 {cfg.cache_path} 加载 {len(sequences)} 条序列")
    else:
        midi_files = load_midi_files(cfg.midi_dir)
        sequences = process_dataset(midi_files)
        if cfg.cache_path:
            with open(cfg.cache_path, "wb") as f:
                pickle.dump(sequences, f)
            print(f"[Data] 已缓存到 {cfg.cache_path}")

    vocab, id_to_token, time_resolution = build_vocab(sequences)
    return sequences, vocab, id_to_token, time_resolution


def create_dataloader(
    sequences: List,
    vocab: Dict[str, int],
    cfg: TrainConfig,
    time_resolution: float,
) -> DataLoader:
    dataset = MidiTokenDataset(
        sequences=sequences,
        vocab=vocab,
        max_len=cfg.max_len,
        time_resolution=time_resolution,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return loader


def train_loop(
    cfg: TrainConfig,
    model: MidiTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    train_loader: DataLoader,
    vocab: Dict[str, int],
    id_to_token: Dict[int, str],
    time_resolution: float,
    start_epoch: int = 0,
    start_step: int = 0,
) -> None:
    device = torch.device(cfg.device)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    use_amp = device.type == "cuda" and cfg.precision in {"fp16", "bf16"}
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(cfg.precision, torch.float32)
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=dtype) if use_amp else contextlib.nullcontext()
    )

    global_step = start_step
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            with autocast_ctx:
                logits = model(input_ids, attention_mask=attention_mask)
                vocab_size = logits.size(-1)
                loss = loss_fn(
                    logits.view(-1, vocab_size), target_ids.view(-1)
                ).view(target_ids.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
                loss = loss / cfg.accumulation_steps

            if scaler and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.accumulation_steps == 0:
                if scaler and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                if scaler and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            if global_step % cfg.save_interval == 0 and global_step > 0:
                save_path = os.path.join(cfg.save_dir, f"midi_transformer_step_{global_step}.pt")
                save_checkpoint(
                    save_path,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    vocab,
                    id_to_token,
                    time_resolution,
                )

            if global_step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[Epoch {epoch+1}/{cfg.epochs}] Step {global_step} "
                    f"Loss {(loss.item() * cfg.accumulation_steps):.4f} LR {lr:.6e}"
                )
            global_step += 1

        spend = (time.time() - epoch_start) / 60
        print(f"[Epoch {epoch+1}] 完成，耗时 {spend:.2f} 分钟")

    # 最终保存
    final_ckpt = os.path.join(cfg.save_dir, "midi_transformer_final.pt")
    save_checkpoint(
        final_ckpt,
        model,
        optimizer,
        scaler,
        cfg.epochs,
        global_step,
        vocab,
        id_to_token,
        time_resolution,
    )


# ============================
# 主入口
# ============================


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="MIDI Transformer (PyTorch) 训练脚本"
    )
    parser.add_argument(
        "--midi_dir",
        type=str,
        default=r"D:\um study\DL\big_project\nottingham-dataset-master\MIDI\melody",
        help="MIDI 数据集目录",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="midi_sequences.pkl",
        help="已处理序列缓存路径",
    )
    parser.add_argument("--max_len", type=int, default=1024, help="token序列最大长度")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="混合精度类型",
    )
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader线程数")
    parser.add_argument("--save_dir", type=str, default="torch_midi_ckpts", help="模型保存目录")
    parser.add_argument(
        "--save_interval",
        type=int,
        default=500,
        help="保存间隔（global step）",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="checkpoint路径，存在则加载",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    return TrainConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    sequences, vocab, id_to_token, time_resolution = prepare_sequences(cfg)
    loader = create_dataloader(sequences, vocab, cfg, time_resolution)

    model = MidiTransformer(
        vocab_size=len(vocab),
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=cfg.max_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    total_steps = cfg.epochs * len(loader) // max(cfg.accumulation_steps, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )
    scaler = (
        torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.precision == "fp16"))
        if device.type == "cuda"
        else None
    )

    start_epoch = 0
    start_step = 0
    if cfg.resume_path and os.path.exists(cfg.resume_path):
        (
            start_epoch,
            start_step,
            vocab,
            id_to_token,
            time_resolution,
        ) = load_checkpoint(cfg.resume_path, model, optimizer, scaler)
        print(
            f"[Resume] 从 epoch {start_epoch} step {start_step} 继续训练；"
            f"词表大小 {len(vocab)}"
        )

    train_loop(
        cfg,
        model,
        optimizer,
        scheduler,
        scaler,
        loader,
        vocab,
        id_to_token,
        time_resolution,
        start_epoch=start_epoch,
        start_step=start_step,
    )


if __name__ == "__main__":
    main()

