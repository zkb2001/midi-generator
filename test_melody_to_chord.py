"""
Melody -> Chord 推理脚本
=========================

用法示例：
    python test_melody_to_chord.py \
        --ckpt_path melody2chord_ckpts/melody2chord_final.pt \
        --melody_midi sample_melody.mid \
        --output_midi predicted_chords.mid \
        --temperature 0.9 --top_k 30
"""

from __future__ import annotations

import argparse
import os
import torch
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from melody_to_chord_transformer import (
    MelodyToChordModel,
    midi_to_events,
    sequence_to_tokens,
    generate_chords,
    tokens_to_events,
    events_to_midi,
)


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MelodyToChordModel(
        vocab_size=len(ckpt["vocab"]),
        d_model=ckpt["cfg"]["d_model"],
        nhead=ckpt["cfg"]["nhead"],
        num_layers=ckpt["cfg"]["num_layers"],
        dim_feedforward=ckpt["cfg"]["dim_feedforward"],
        dropout=ckpt["cfg"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def events_to_track(events, track, tempo_bpm=120, ticks_per_beat=480):
    """将事件列表添加到 MIDI track"""
    tempo_microseconds = mido.bpm2tempo(tempo_bpm)
    for event in events:
        if len(event) == 2:
            note, duration = event
            velocity = 64
        elif len(event) == 4:
            note, duration, _, velocity = event
        else:
            continue
        
        track.append(Message("note_on", note=int(note), velocity=velocity, time=0))
        ticks_duration = int(mido.second2tick(duration, ticks_per_beat, tempo_microseconds))
        track.append(Message("note_off", note=int(note), velocity=0, time=max(ticks_duration, 1)))


def merge_melody_chord_midi(
    melody_events,
    chord_events,
    output_path: str,
    tempo_bpm: int = 120,
    ticks_per_beat: int = 480,
):
    """将 melody 和 chord 事件合并到一个 MIDI 文件中（两个 track）"""
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    tempo_microseconds = mido.bpm2tempo(tempo_bpm)
    
    # Melody track
    melody_track = MidiTrack()
    melody_track.append(MetaMessage("set_tempo", tempo=tempo_microseconds, time=0))
    events_to_track(melody_events, melody_track, tempo_bpm=tempo_bpm, ticks_per_beat=ticks_per_beat)
    mid.tracks.append(melody_track)
    
    # Chord track
    chord_track = MidiTrack()
    chord_track.append(MetaMessage("set_tempo", tempo=tempo_microseconds, time=0))
    events_to_track(chord_events, chord_track, tempo_bpm=tempo_bpm, ticks_per_beat=ticks_per_beat)
    mid.tracks.append(chord_track)
    
    mid.save(output_path)
    print(f"[MIDI] 合并文件已保存到 {output_path}")


def run_inference(args):
    device = torch.device(args.device)
    model, ckpt = load_checkpoint(args.ckpt_path, device)
    vocab = ckpt["vocab"]
    id_to_token = ckpt["id_to_token"]
    time_resolution = ckpt["time_resolution"]
    max_src_len = ckpt["cfg"]["max_src_len"]
    max_tgt_len = ckpt["cfg"]["max_tgt_len"]

    melody_events = midi_to_events(args.melody_midi)
    if melody_events is None:
        raise RuntimeError(f"无法解析 melody MIDI: {args.melody_midi}")

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
    chord_events = tokens_to_events(token_ids, id_to_token, time_resolution)
    if not chord_events:
        raise RuntimeError("未生成任何 chord 事件")

    # 输出目录准备
    output_dir = os.path.dirname(args.output_midi) or "."
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 输出单独的 chord MIDI
    chord_only_path = args.output_midi
    events_to_midi(chord_events, chord_only_path)
    print(f"[Done] 生成 {len(chord_events)} 个 chord 事件，单独输出至 {chord_only_path}")
    
    # 2. 输出合并后的完整 MIDI（melody + chord）
    base_name = os.path.splitext(args.output_midi)[0]
    merged_path = f"{base_name}_merged.mid"
    merge_melody_chord_midi(melody_events, chord_events, merged_path)
    print(f"[Done] 合并文件（melody + chord）已保存至 {merged_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Melody -> Chord 推理脚本")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="melody2chord_ckpts/melody2chord_final.pt",
        help="训练好的 checkpoint",
    )
    parser.add_argument(
        "--melody_midi",
        type=str,
        default=None,
        help="输入 melody MIDI 文件路径",
    )
    parser.add_argument("--output_midi", type=str, default="predicted_chords.mid")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    
    # 如果未提供 melody_midi，提示用户输入
    if args.melody_midi is None:
        args.melody_midi = input("请输入 melody MIDI 文件路径: ").strip()
        if not args.melody_midi:
            raise ValueError("必须提供 melody MIDI 文件路径")
    
    return args


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

