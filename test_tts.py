"""Quick Kokoro TTS test."""
import os
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

from kokoro import KPipeline
import soundfile as sf
from pathlib import Path

OUTPUT = Path(r"E:\Rexjaw_VFX\04_Outputs\tts_test.wav")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

pipeline = KPipeline(lang_code='a')   # 'a' = American English
text = "Once upon a time, a tiny dragon was afraid of his own fire. But with help from his friends, he learned that fire could warm hearts, not just burn things."
voice = "af_heart"   # warm female narrator

print(f"Generating with voice: {voice}")
audio_chunks = []
for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice)):
    print(f"  Chunk {i}: '{gs[:50]}...'")
    audio_chunks.append(audio)

import numpy as np
combined = np.concatenate(audio_chunks)
sf.write(str(OUTPUT), combined, 24000)
print(f"\nSaved: {OUTPUT}")
print(f"Duration: {len(combined)/24000:.2f} seconds")