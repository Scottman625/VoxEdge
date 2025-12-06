from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.generators import WhiteNoise
import random

# --- 設定 ---
COMMANDS = {
    "kai": ["開"],
    "qi": ["啟" ],
    "guan": ["關"],
    "bi": ["閉" ],
    "silence": ["", " ", "   "] # 這裡會產生噪音
}

OUTPUT_DIR = "dataset"
SPEEDS = [0.85, 0.95, 1.0, 1.1, 1.2]
PITCH_SEMITONES = [-2, 0, 2]
SILENCE_DURATION_MS = 512 
TARGET_DURATION_MS = 512
TTS_ENGINES = ["gtts", "edge"]
TTS_LANG = "zh-tw"
EDGE_VOICE = "zh-TW-YunJheNeural"

def _augment_and_export(base_sound: AudioSegment, out_path: str, speed: float = 1.0, semitones: int = 0):
    # ... (這部分保持不變) ...
    factor = speed * (2.0 ** (semitones / 12.0))
    new_frame_rate = int(base_sound.frame_rate * factor)
    modified = base_sound._spawn(base_sound.raw_data, overrides={"frame_rate": new_frame_rate})
    modified = modified.set_frame_rate(16000).set_channels(1)
    
    if len(modified) > TARGET_DURATION_MS:
        modified = modified[:TARGET_DURATION_MS]
    elif len(modified) < TARGET_DURATION_MS:
        pad_len = TARGET_DURATION_MS - len(modified)
        silence = AudioSegment.silent(duration=pad_len)
        modified = modified + silence
    
    modified.export(out_path, format="wav")

def generate_audio():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("🤖 開始生成合成語音並做擴增...")

    for label, phrases in COMMANDS.items():
        label_dir = os.path.join(OUTPUT_DIR, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        for i, phrase in enumerate(phrases):
            variant_total = 0

            # ⚠️ 修改：如果是 Silence，產生「噪音」而不是「靜音」
            if label == "silence":
                # 產生白噪音，音量設小一點 (-30dB ~ -40dB)
                noise_gen = WhiteNoise()
                # 產生幾個不同音量的變體
                for vol_adj in [-30, -40, -50]: 
                    base_sound = noise_gen.to_audio_segment(duration=SILENCE_DURATION_MS, volume=vol_adj)
                    base_sound = base_sound.set_frame_rate(16000).set_channels(1)
                    
                    out_fname = f"noise_gen_{i}_vol{vol_adj}.wav"
                    out_path = os.path.join(label_dir, out_fname)
                    base_sound.export(out_path, format="wav")
                    variant_total += 1

                print(f"   ➕ 生成噪音: (silence) -> {label} ({variant_total} variants)")
                continue

            # ... (TTS 生成部分保持不變) ...
            for engine in TTS_ENGINES:
                temp_mp3 = os.path.join(label_dir, f"tts_{i}_{engine}.mp3")
                
                # (省略中間 TTS 程式碼，請保持原樣)
                if engine == "gtts":
                    try:
                        tts = gTTS(text=phrase, lang=TTS_LANG, slow=False)
                        tts.save(temp_mp3)
                    except: continue
                elif engine == "edge":
                    try:
                        import asyncio
                        import edge_tts
                        async def _save_edge(text, voice, path):
                            communicate = edge_tts.Communicate(text, voice=voice)
                            await communicate.save(path)
                        asyncio.run(_save_edge(phrase, EDGE_VOICE, temp_mp3))
                    except: continue
                
                try:
                    base_sound = AudioSegment.from_file(temp_mp3)
                except: continue

                variant_idx = 0
                for speed in SPEEDS:
                    for semitone in PITCH_SEMITONES:
                        out_fname = f"{engine}_tts_{i}_v{variant_idx}_s{int(speed*100)}_p{semitone}.wav"
                        out_path = os.path.join(label_dir, out_fname)
                        _augment_and_export(base_sound, out_path, speed=speed, semitones=semitone)
                        variant_idx += 1
                        variant_total += 1
                try: os.remove(temp_mp3)
                except: pass
            
            print(f"   ➕ 生成並擴增: '{phrase}' -> {label} ({variant_total} variants total)")

    print("✅ 資料生成完畢！")

if __name__ == "__main__":
    generate_audio()
