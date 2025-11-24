"""
交互式 MIDI 生成脚本：运行后可按提示输入参数（或直接回车使用默认值）。

示例：
    python test_torch_midi_model_custom.py
    -> 依次输入 checkpoint 路径、输出文件、max_events 等
"""

from __future__ import annotations

import os
import torch

from test_torch_midi_model import (
    DEFAULT_CONFIG,
    events_to_midi,
    load_model,
    sample_next_token,
    tokens_to_events,
)
def prepare_seed_sequence(
    vocab,
    note: int,
    duration: float,
    time_resolution: float,
) -> list[int]:
    note_token = vocab.get(f"note_{int(note)}")
    dur_idx = max(int(round(duration / time_resolution)), 0)
    dur_token = vocab.get(f"dur_{dur_idx}")
    if note_token is None or dur_token is None:
        raise ValueError("输入的音高/时长不在词表中，请调整")
    return [vocab["<sos>"], note_token, dur_token]


def generate_with_seed(
    model,
    vocab,
    id_to_token,
    seed_tokens,
    max_events: int,
    device: torch.device,
    temperature: float,
    top_k: int,
    max_len: int,
):
    generated = seed_tokens[:]
    format_state = 0 if len(seed_tokens) % 2 == 1 else 0
    events = (len(seed_tokens) - 1) // 2

    with torch.no_grad():
        while len(generated) < max_len:
            input_ids = torch.tensor(generated, device=device).unsqueeze(0)
            attn_mask = torch.ones_like(input_ids)
            logits = model(input_ids, attention_mask=attn_mask)
            next_logits = logits[0, -1]

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


def prompt(value: str, default: str) -> str:
    user_input = input(f"{value} [默认: {default}]: ").strip()
    return user_input or default


def main():
    print("=== MIDI Transformer 交互式生成 ===")
    base_cfg = DEFAULT_CONFIG.copy()
    ckpt_path = base_cfg["ckpt_path"]
    if not os.path.exists(ckpt_path):
        print(f"[Error] 找不到 {ckpt_path}")
        return
    output_midi = base_cfg["output_midi"]
    max_events = base_cfg["max_events"]
    temperature = base_cfg["temperature"]
    top_k = base_cfg["top_k"]
    max_len = base_cfg["max_len"]

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)
    model, vocab, id_to_token, time_resolution = load_model(
        ckpt_path,
        device,
        max_len=max_len,
    )

    custom_start = input("是否自定义起始音符? (y/N): ").strip().lower() == "y"
    if custom_start:
        note = float(input("请输入起始音高 (0-127): ").strip() or "60")
        duration = float(input("请输入持续时长(秒): ").strip() or "0.5")
        seed_tokens = prepare_seed_sequence(vocab, note, duration, time_resolution)
        tokens = generate_with_seed(
            model,
            vocab,
            id_to_token,
            seed_tokens,
            max_events=max_events,
            device=device,
            temperature=temperature,
            top_k=top_k,
            max_len=max_len,
        )
    else:
        from test_torch_midi_model import generate_tokens

        tokens = generate_tokens(
            model,
            vocab,
            id_to_token,
            max_events=max_events,
            device=device,
            temperature=temperature,
            top_k=top_k,
            max_len=max_len,
        )
    events = tokens_to_events(tokens, id_to_token, time_resolution)
    print(f"[Info] 生成 {len(tokens)} 个 token，对应 {len(events)} 个事件")
    if len(events) == 0:
        print("[Warn] 未生成事件，请调整参数重试")
        return

    os.makedirs(os.path.dirname(output_midi) or ".", exist_ok=True)
    events_to_midi(events, output_midi)
    print(f"[Done] 已将 {len(events)} 个事件写入 {output_midi}")


if __name__ == "__main__":
    main()

# 示例命令：
# python test_torch_midi_model_custom.py
# （根据提示输入或直接回车使用默认值）

