"""
下载MIDI数据集
提供多个数据源选项
"""
import urllib.request
import os
import zipfile
import shutil


def download_maestro():
    """下载MAESTRO数据集（需要手动下载）"""
    print("=" * 60)
    print("MAESTRO数据集下载说明")
    print("=" * 60)
    print("\nMAESTRO是一个高质量的钢琴音乐数据集")
    print("下载地址: https://magenta.tensorflow.org/datasets/maestro")
    print("\n或者直接下载:")
    print("wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip")
    print("\n下载后解压到 midi_data/maestro/ 目录")
    return None


def download_sample_midi():
    """下载示例MIDI文件（用于测试）"""
    data_dir = "midi_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 一些公开的MIDI文件URL（示例）
    sample_urls = [
        # 可以添加一些公开的MIDI文件URL
    ]
    
    print("提示：可以从以下网站下载免费MIDI文件：")
    print("1. https://www.midiworld.com/")
    print("2. https://freemidi.org/")
    print("3. https://bitmidi.com/")
    print("\n下载后放入 midi_data/ 目录即可")


def create_sample_midi():
    """创建一些示例MIDI文件用于测试"""
    try:
        import mido
        from mido import MidiFile, MidiTrack, Message
        
        data_dir = "midi_data/samples"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 创建简单的C大调音阶
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # C大调音阶: C, D, E, F, G, A, B, C
        notes = [60, 62, 64, 65, 67, 69, 71, 72]
        
        for note in notes:
            track.append(Message('note_on', note=note, velocity=64, time=0))
            track.append(Message('note_off', note=note, velocity=64, time=480))
        
        mid.save(os.path.join(data_dir, 'scale_c_major.mid'))
        print(f"创建示例MIDI文件: {data_dir}/scale_c_major.mid")
        
        # 创建更多示例
        for i in range(5):
            mid = MidiFile()
            track = MidiTrack()
            mid.tracks.append(track)
            
            # 随机生成简单旋律
            for _ in range(8):
                note = random.randint(60, 72)
                track.append(Message('note_on', note=note, velocity=64, time=0))
                track.append(Message('note_off', note=note, velocity=64, time=480))
            
            mid.save(os.path.join(data_dir, f'sample_{i+1}.mid'))
        
        print(f"创建了6个示例MIDI文件在 {data_dir}/")
        return data_dir
        
    except ImportError:
        print("需要安装mido库: pip install mido")
        return None


def main():
    """主函数"""
    print("=" * 60)
    print("MIDI数据集准备工具")
    print("=" * 60)
    
    print("\n选项:")
    print("1. 创建示例MIDI文件（用于测试）")
    print("2. 查看MAESTRO数据集下载说明")
    print("3. 查看其他MIDI数据源")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == '1':
        import random
        create_sample_midi()
    elif choice == '2':
        download_maestro()
    elif choice == '3':
        download_sample_midi()
    else:
        print("创建示例MIDI文件...")
        import random
        create_sample_midi()


if __name__ == "__main__":
    main()

