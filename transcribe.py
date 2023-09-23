import glob
import os
import argparse

from faster_whisper import WhisperModel
from tqdm import tqdm


def transcribe(speaker):
   model_size = "large-v2"
   model = WhisperModel(model_size, device="cuda", compute_type="float16")
   audio = glob.glob("dataset/*.wav")
   text_list = []
   for i in tqdm(audio):
      sound_file = os.path.basename(i)
      segments, info = model.transcribe(i, beam_size=5, language="ja")
      text = ""
      for segment in segments:
         text += segment.text
      text_list.append(f"wav/{args.speaker}/{sound_file}|{text}")
   with open("dataset/voice_text.txt", "w", encoding='utf-8')as f:
      for i in text_list:
         f.write(i + "\n")


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--speaker", default="speaker")
   args = parser.parse_args()
   transcribe(args.speaker)
