"""
Melody-to-Chord Transformer
===========================

使用 melody MIDI 预测 chord MIDI：
1. 读取 melody 与 chord 目录（按排序后逐一配对）。
2. 将每首曲子的事件序列量化为 note/duration token。
3. 训练一个 Transformer 编码器-解码器模型：Encoder 读取 melody，Decoder 生成 chord。
4. 支持推理：输入单独的 melody MIDI，输出对应的 chord MIDI。
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from process_midi_simple import (
    midi_to_events,
    build_vocab,
    sequence_to_tokens,
    dequantize_time,
    events_to_midi,
)


def list_midis(directory: str) -> List[str]:
    return sorted(
        [
            os.path.join(root, name)
            for root, _, files in os.walk(directory)
            for name in files
            if name.lower().endswith((".mid", ".midi"))
        ]
    )


def load_paired_sequences(
    melody_dir: str, chord_dir: str
) -> List[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]:
    melody_files = list_midis(melody_dir)
    chord_files = list_midis(chord_dir)
    if len(melody_files) == 0 or len(chord_files) == 0:
        raise RuntimeError("melody/chord 目录中未找到 MIDI 文件")

    def index_by_name(paths: List[str]):
        mapping = {}
        for path in paths:
            name = os.path.basename(path)
            mapping.setdefault(name, []).append(path)
        return mapping

    melody_map = index_by_name(melody_files)
    chord_map = index_by_name(chord_files)

    melody_names = set(melody_map.keys())
    chord_names = set(chord_map.keys())
    common_names = sorted(melody_names & chord_names)
    missing_in_chord = sorted(melody_names - chord_names)
    missing_in_melody = sorted(chord_names - melody_names)

    if missing_in_chord:
        print("[Warn] 以下 melody 文件在 chord 目录中缺失:")
        for name in missing_in_chord:
            print(f"  - {name}")
    if missing_in_melody:
        print("[Warn] 以下 chord 文件在 melody 目录中缺失:")
        for name in missing_in_melody:
            print(f"  - {name}")

    if not common_names:
        raise RuntimeError("两目录无同名 MIDI 文件，无法配对")

    pairs: List[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]] = []
    for name in common_names:
        melody_path = melody_map[name][0]
        chord_path = chord_map[name][0]
        melody_events = midi_to_events(melody_path)
        chord_events = midi_to_events(chord_path)
        if melody_events is None or chord_events is None:
            continue
        pairs.append((melody_events, chord_events))
    if not pairs:
        raise RuntimeError("未能解析任何 melody-chord 配对")
    print(f"[Data] 共找到 {len(common_names)} 对同名 MIDI，其中成功解析 {len(pairs)} 对")
    return pairs


@dataclass
class TrainConfig:
    melody_dir: str
    chord_dir: str
    save_dir: str
    batch_size: int
    epochs: int
    lr: float
    max_src_len: int
    max_tgt_len: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    device: str
    time_resolution: float
    max_duration: float
    val_ratio: float


class MelodyChordDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]],
        vocab,
        max_src_len: int,
        max_tgt_len: int,
        time_resolution: float,
    ):
        self.pad_id = vocab["<pad>"]
        self.src_sequences = []
        self.tgt_in_sequences = []
        self.tgt_out_sequences = []

        for melody_events, chord_events in pairs:
            src_tokens = sequence_to_tokens(
                melody_events,
                vocab,
                max_len=max_src_len,
                time_resolution=time_resolution,
            )
            tgt_tokens = sequence_to_tokens(
                chord_events,
                vocab,
                max_len=max_tgt_len,
                time_resolution=time_resolution,
            )
            tgt_input = tgt_tokens[:-1]
            tgt_output = tgt_tokens[1:]

            self.src_sequences.append(torch.from_numpy(src_tokens).long())
            self.tgt_in_sequences.append(torch.from_numpy(tgt_input).long())
            self.tgt_out_sequences.append(torch.from_numpy(tgt_output).long())

    def __len__(self) -> int:
        return len(self.src_sequences)

    def __getitem__(self, idx: int):
        return {
            "src": self.src_sequences[idx],
            "tgt_in": self.tgt_in_sequences[idx],
            "tgt_out": self.tgt_out_sequences[idx],
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MelodyToChordModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        src_key_padding_mask: torch.Tensor | None,
        tgt_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )
        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_proj(output)


def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.full((size, size), float("-inf"), device=device), diagonal=1
    )


def tokens_to_events(tokens, id_to_token, time_resolution):
    events = []
    i = 0
    while i < len(tokens) - 1:
        token = id_to_token.get(int(tokens[i]), "")
        if token.startswith("note_") and i + 1 < len(tokens):
            note = int(token.split("_")[1])
            dur_token = id_to_token.get(int(tokens[i + 1]), "")
            if dur_token.startswith("dur_"):
                dur_idx = int(dur_token.split("_")[1])
                duration = dequantize_time(dur_idx, time_resolution)
            else:
                duration = 0.5
            events.append((note, duration))
            i += 2
        else:
            i += 1
    return events


def train_loop(cfg: TrainConfig) -> None:
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    pairs = load_paired_sequences(cfg.melody_dir, cfg.chord_dir)
    random.shuffle(pairs)
    val_ratio = min(max(cfg.val_ratio, 0.0), 0.5)
    if val_ratio <= 0.0 or len(pairs) < 2:
        train_pairs = pairs
        val_pairs: List[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]] = []
    else:
        split_idx = max(1, min(len(pairs) - 1, int(len(pairs) * (1 - val_ratio))))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
        print(
            f"[Data] 使用 {len(train_pairs)} 对作为训练，{len(val_pairs)} 对作为验证 (ratio={val_ratio:.2f})"
        )

    combined_sequences = [seq for pair in pairs for seq in pair]
    vocab, id_to_token, time_resolution = build_vocab(
        combined_sequences,
        time_resolution=cfg.time_resolution,
        max_duration=cfg.max_duration,
    )

    dataset = MelodyChordDataset(
        train_pairs,
        vocab,
        max_src_len=cfg.max_src_len,
        max_tgt_len=cfg.max_tgt_len,
        time_resolution=time_resolution,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = None
    if val_pairs:
        val_dataset = MelodyChordDataset(
            val_pairs,
            vocab,
            max_src_len=cfg.max_src_len,
            max_tgt_len=cfg.max_tgt_len,
            time_resolution=time_resolution,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )

    model = MelodyToChordModel(
        vocab_size=len(vocab),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)

            src_pad = src == vocab["<pad>"]
            tgt_pad = tgt_in == vocab["<pad>"]
            tgt_mask = subsequent_mask(tgt_in.size(1), device)

            logits = model(
                src,
                tgt_in,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_pad,
                tgt_key_padding_mask=tgt_pad,
            )
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    src = batch["src"].to(device)
                    tgt_in = batch["tgt_in"].to(device)
                    tgt_out = batch["tgt_out"].to(device)

                    src_pad = src == vocab["<pad>"]
                    tgt_pad = tgt_in == vocab["<pad>"]
                    tgt_mask = subsequent_mask(tgt_in.size(1), device)

                    logits = model(
                        src,
                        tgt_in,
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_pad,
                        tgt_key_padding_mask=tgt_pad,
                    )
                    loss = loss_fn(
                        logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
                    )
                    val_loss += loss.item()
            val_loss = val_loss / max(len(val_loader), 1)
            print(
                f"[Epoch {epoch+1}/{cfg.epochs}] TrainLoss {avg_loss:.4f} | ValLoss {val_loss:.4f}"
            )
        else:
            print(f"[Epoch {epoch+1}/{cfg.epochs}] Loss {avg_loss:.4f}")

    ckpt = {
        "model": model.state_dict(),
        "vocab": vocab,
        "id_to_token": id_to_token,
        "time_resolution": time_resolution,
        "cfg": cfg.__dict__,
    }
    save_path = os.path.join(cfg.save_dir, "melody2chord_final.pt")
    torch.save(ckpt, save_path)
    print(f"[Checkpoint] 已保存到 {save_path}")


def sample_next_token(logits, temperature: float, top_k: int) -> int:
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return indices[choice].item()
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate_chords(
    model: MelodyToChordModel,
    melody_tokens,
    vocab,
    id_to_token,
    max_len: int,
    device: torch.device,
    temperature: float,
    top_k: int,
    melody_max_events: int | None = None,
    time_resolution: float | None = None,
) -> List[int]:
    """
    生成 chord tokens
    
    Args:
        melody_max_events: melody 的实际事件数量（用于限制 chord 生成长度）
        time_resolution: 时间分辨率（用于基于时长的限制）
    """
    model.eval()
    src = torch.tensor(melody_tokens, device=device).unsqueeze(0)
    src_pad = src == vocab["<pad>"]
    generated = [vocab["<sos>"]]
    format_state = 0
    event_count = 0
    total_duration = 0.0
    melody_total_duration = None

    # 计算 melody 的实际事件数量和总时长（如果未提供）
    if melody_max_events is None:
        # 从 melody_tokens 中提取实际事件（去除 padding 和特殊标记）
        melody_actual_events = 0
        melody_total_duration = 0.0
        i = 1  # 跳过 <sos>
        while i < len(melody_tokens) - 1:
            token = id_to_token.get(int(melody_tokens[i]), "")
            if token.startswith("note_") and i + 1 < len(melody_tokens):
                dur_token = id_to_token.get(int(melody_tokens[i + 1]), "")
                if dur_token.startswith("dur_"):
                    dur_idx = int(dur_token.split("_")[1])
                    if time_resolution is not None:
                        duration = dequantize_time(dur_idx, time_resolution)
                        melody_total_duration += duration
                    melody_actual_events += 1
                    i += 2
                    continue
            if token in (vocab.get("<pad>"), vocab.get("<eos>")):
                break
            i += 1
        melody_max_events = melody_actual_events
    else:
        # 如果提供了 melody_max_events，仍然需要计算总时长用于时长限制
        if time_resolution is not None:
            melody_total_duration = 0.0
            i = 1  # 跳过 <sos>
            while i < len(melody_tokens) - 1:
                token = id_to_token.get(int(melody_tokens[i]), "")
                if token.startswith("note_") and i + 1 < len(melody_tokens):
                    dur_token = id_to_token.get(int(melody_tokens[i + 1]), "")
                    if dur_token.startswith("dur_"):
                        dur_idx = int(dur_token.split("_")[1])
                        duration = dequantize_time(dur_idx, time_resolution)
                        melody_total_duration += duration
                        i += 2
                        continue
                if token in (vocab.get("<pad>"), vocab.get("<eos>")):
                    break
                i += 1

    # 限制生成的事件数量：不超过 melody 事件数量的 1.2 倍
    max_events = min(max_len, int(melody_max_events * 1.2) if melody_max_events else max_len)

    with torch.no_grad():
        for _ in range(max_len * 2):
            tgt = torch.tensor(generated, device=device).unsqueeze(0)
            tgt_mask = subsequent_mask(tgt.size(1), device)
            logits = model(
                src,
                tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_pad,
                tgt_key_padding_mask=None,
            )
            next_logits = logits[0, -1]

            filtered = torch.full_like(next_logits, float("-inf"))
            if format_state == 0:
                for note in range(128):
                    token = f"note_{note}"
                    if token in vocab:
                        filtered[vocab[token]] = next_logits[vocab[token]]
            else:
                for dur_idx in range(101):
                    token = f"dur_{dur_idx}"
                    if token in vocab:
                        filtered[vocab[token]] = next_logits[vocab[token]]
            if torch.isinf(filtered).all():
                filtered = next_logits

            next_id = sample_next_token(filtered, temperature, top_k)
            generated.append(next_id)

            token = id_to_token.get(next_id, "")
            if format_state == 0 and token.startswith("note_"):
                format_state = 1
            elif format_state == 1 and token.startswith("dur_"):
                format_state = 0
                event_count += 1
                # 基于事件数量限制
                if melody_max_events and event_count >= max_events:
                    break
                # 基于时长限制
                if time_resolution and melody_total_duration:
                    dur_idx = int(token.split("_")[1])
                    duration = dequantize_time(dur_idx, time_resolution)
                    total_duration += duration
                    if total_duration >= melody_total_duration * 1.1:  # 允许10%的误差
                        break
            if next_id == vocab["<eos>"]:
                break
            if len(generated) >= max_len:
                break
    return generated


def run_inference(args):
    device = torch.device(args.device)
    if not args.checkpoint:
        raise ValueError("推理模式需要提供 --checkpoint")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = MelodyToChordModel(
        vocab_size=len(ckpt["vocab"]),
        d_model=ckpt["cfg"]["d_model"],
        nhead=ckpt["cfg"]["nhead"],
        num_layers=ckpt["cfg"]["num_layers"],
        dim_feedforward=ckpt["cfg"]["dim_feedforward"],
        dropout=ckpt["cfg"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    vocab = ckpt["vocab"]
    id_to_token = ckpt["id_to_token"]
    time_resolution = ckpt["time_resolution"]
    max_src_len = ckpt["cfg"]["max_src_len"]
    max_tgt_len = ckpt["cfg"]["max_tgt_len"]

    melody_events = midi_to_events(args.infer_melody)
    if melody_events is None:
        raise RuntimeError("无法解析输入 melody MIDI")
    melody_tokens = sequence_to_tokens(
        melody_events,
        vocab,
        max_len=max_src_len,
        time_resolution=time_resolution,
    )

    # 计算 melody 的实际事件数量（用于限制 chord 生成长度）
    melody_actual_events = len(melody_events)

    token_ids = generate_chords(
        model,
        melody_tokens,
        vocab,
        id_to_token,
        max_len=max_tgt_len,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        melody_max_events=melody_actual_events,
        time_resolution=time_resolution,
    )
    events = tokens_to_events(token_ids, id_to_token, time_resolution)
    if not events:
        raise RuntimeError("未生成任何和弦事件")

    os.makedirs(os.path.dirname(args.output_midi) or ".", exist_ok=True)
    events_to_midi(events, args.output_midi)
    print(f"[Done] 已生成 {len(events)} 个 chord 事件 -> {args.output_midi}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Melody -> Chord Transformer 训练与推理脚本"
    )
    parser.add_argument(
        "--melody_dir",
        type=str,
        default=r"D:\um study\DL\midi_generator\nottingham-dataset-master\MIDI\melody",
        help="melody MIDI 目录",
    )
    parser.add_argument(
        "--chord_dir",
        type=str,
        default=r"D:\um study\DL\midi_generator\nottingham-dataset-master\MIDI\chords",
        help="chord MIDI 目录",
    )
    parser.add_argument("--save_dir", type=str, default="melody2chord_ckpts")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_src_len", type=int, default=2048)
    parser.add_argument("--max_tgt_len", type=int, default=2048)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--time_resolution", type=float, default=0.05)
    parser.add_argument("--max_duration", type=float, default=5.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--infer_melody", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_midi", type=str, default="predicted_chords.mid")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例 (0-0.5)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.infer_melody:
        run_inference(args)
        return
    cfg = TrainConfig(
        melody_dir=args.melody_dir,
        chord_dir=args.chord_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        device=args.device,
        time_resolution=args.time_resolution,
        max_duration=args.max_duration,
        val_ratio=args.val_ratio,
    )
    train_loop(cfg)


if __name__ == "__main__":
    main()

