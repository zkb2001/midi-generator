"""
使用已训练的 PyTorch MIDI Transformer (RoPE) 生成测试样本。

最简单方式：直接运行/双击该文件，脚本会自动使用默认配置：
    - checkpoint: torch_midi_ckpts/midi_transformer_final.pt
    - 输出: generated_rope.mid
    - max_events=80, temperature=0.9, top_k=50

如需自定义，继续通过命令行参数覆盖即可，例如：
    python test_torch_midi_model.py --output_midi demo.mid --max_events 64
"""

import argparse
import os
import torch

from torch_midi_transformer import MidiTransformer
from process_midi_simple import events_to_midi, dequantize_time


DEFAULT_CONFIG = {
    "ckpt_path": "torch_midi_ckpts/midi_transformer_final.pt",
    "output_midi": "generated_rope.mid",
    "max_events": 80,
    "temperature": 0.9,
    "top_k": 50,
    "max_len": 2048,
}


def load_model(ckpt_path: str, device: torch.device, **model_kwargs):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
    id_to_token = ckpt["id_to_token"]
    time_resolution = ckpt["time_resolution"]

    model = MidiTransformer(
        vocab_size=len(vocab),
        d_model=model_kwargs.get("d_model", 256),
        nhead=model_kwargs.get("nhead", 8),
        num_layers=model_kwargs.get("num_layers", 4),
        dim_feedforward=model_kwargs.get("dim_feedforward", 1024),
        dropout=model_kwargs.get("dropout", 0.1),
        max_len=model_kwargs.get("max_len", 1024),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"[Load] ckpt={ckpt_path}, epoch={ckpt.get('epoch')}, step={ckpt.get('step')}, vocab={len(vocab)}"
    )
    return model, vocab, id_to_token, time_resolution


def sample_next_token(logits, temperature: float, top_k: int):
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
        probs = torch.softmax(values, dim=-1)
        next_token = indices[torch.multinomial(probs, num_samples=1)]
        return next_token.item()
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate_tokens(
    model,
    vocab,
    id_to_token,
    max_events: int,
    device: torch.device,
    temperature: float,
    top_k: int,
    max_len: int,
):
    generated = [vocab["<sos>"]]
    format_state = 0  # 0: note, 1: duration
    events = 0

    with torch.no_grad():
        while len(generated) < max_len:
            input_ids = torch.tensor(generated, device=device).unsqueeze(0)
            attn_mask = torch.ones_like(input_ids)
            logits = model(input_ids, attention_mask=attn_mask)
            next_logits = logits[0, -1]

            # 限制 token 类型（note 与 duration 交替）
            filtered_logits = torch.full_like(next_logits, -1e9)
            if format_state == 0:
                for note in range(128):
                    token = f"note_{note}"
                    if token in vocab:
                        filtered_logits[vocab[token]] = next_logits[vocab[token]]
            else:
                for dur_idx in range(101):
                    token = f"dur_{dur_idx}"
                    if token in vocab:
                        filtered_logits[vocab[token]] = next_logits[vocab[token]]
            next_id = sample_next_token(filtered_logits, temperature, top_k)
            generated.append(next_id)

            token = id_to_token.get(next_id, "")
            if format_state == 0 and token.startswith("note_"):
                format_state = 1
            elif format_state == 1 and token.startswith("dur_"):
                format_state = 0
                events += 1
                if events >= max_events:
                    break

            if next_id == vocab["<eos>"]:
                break
    return generated


def tokens_to_events(tokens, id_to_token, time_resolution):
    events = []
    i = 1  # skip <sos>
    while i < len(tokens) - 1:
        token = id_to_token.get(tokens[i], "")
        if token.startswith("note_") and i + 1 < len(tokens):
            note = int(token.split("_")[1])
            dur_token = id_to_token.get(tokens[i + 1], "")
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


def main():
    parser = argparse.ArgumentParser(description="MIDI Transformer 生成测试脚本")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=DEFAULT_CONFIG["ckpt_path"],
        help="checkpoint 路径",
    )
    parser.add_argument(
        "--output_midi",
        type=str,
        default=DEFAULT_CONFIG["output_midi"],
        help="生成 MIDI 文件路径",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=DEFAULT_CONFIG["max_events"],
        help="最多生成多少个 (note, duration) 事件",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_CONFIG["temperature"],
        help="softmax 温度",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_CONFIG["top_k"],
        help="top-k 采样范围（<=0 表示不裁剪）",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=DEFAULT_CONFIG["max_len"],
        help="生成序列允许的最大 token 长度",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, vocab, id_to_token, time_resolution = load_model(
        args.ckpt_path,
        device,
        max_len=args.max_len,
    )

    tokens = generate_tokens(
        model,
        vocab,
        id_to_token,
        max_events=args.max_events,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        max_len=args.max_len,
    )
    events = tokens_to_events(tokens, id_to_token, time_resolution)
    if len(events) == 0:
        print("未生成有效事件")
        return

    os.makedirs(os.path.dirname(args.output_midi) or ".", exist_ok=True)
    events_to_midi(events, args.output_midi)
    print(f"[Done] 生成 {len(events)} 个事件，已保存到 {args.output_midi}")


if __name__ == "__main__":
    main()

