import argparse
import glob
import json
import os
import shutil

import librosa
import soundfile as sf
from tqdm import tqdm

import text
from utils import load_filepaths_and_text


def resample(speaker):
   wavs = glob.glob("dataset/*.wav")
   if not os.path.exists(f"wav/{speaker}"):
      os.mkdir(f"wav/{speaker}")
   for wavfile in tqdm(wavs):
      y, sr = librosa.load(wavfile, sr=22050, mono=True)
      name = os.path.basename(wavfile)
      sf.write(f"wav/{speaker}/{name}", y, sr, subtype="PCM_16")


def make_config(speaker):
   with open("configs/config.json", "r") as f:
      config = json.load(f)
   config["data"]["training_files"] = f"filelists/{speaker}/train.txt.cleaned"
   config["data"]["validation_files"] = f"filelists/{speaker}/val.txt.cleaned"
   config["speakers"] = [speaker]
   with open(f"configs/{speaker}.json", "w") as f:
      json.dump(config, f, indent=3)


def val(speaker):
   with open("dataset/voice_text.txt", encoding='utf-8')as f:
      speaker_text = f.read().splitlines()
   train = speaker_text[3::]
   val = speaker_text[0:3]
   with open("dataset/train.txt", "w", encoding='utf-8')as f:
      for i in train:
         f.write(i + "\n")

   with open("dataset/val.txt", "w", encoding='utf-8')as f:
      for i in val:
         f.write(i + "\n")
   if not os.path.exists(f"filelists/{speaker}"):
      os.mkdir(f"filelists/{speaker}")
   shutil.copy("dataset/train.txt", f"filelists/{speaker}/train.txt")
   shutil.copy("dataset/val.txt", f"filelists/{speaker}/val.txt")


def preprocess(speaker):
   filelists = [f"filelists/{speaker}/train.txt", f"filelists/{speaker}/val.txt"]
   for filelist in filelists:
      filepaths_and_text = load_filepaths_and_text(filelist)
      for i in range(len(filepaths_and_text)):
         original_text = filepaths_and_text[i][1]
         cleaned_text = text._clean_text(original_text, ["japanese_cleaners"])
         filepaths_and_text[i][1] = cleaned_text
      new_filelist = filelist + "." + "cleaned"
      with open(new_filelist, "w", encoding="utf-8") as f:
         f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])


def make_model(speaker):
   if os.path.exists(f"save_models/{speaker}") is False:
      os.mkdir(f"save_models/{speaker}")
   shutil.copy("save_models/D_0.pth", f"save_models/{speaker}/D_0.pth")
   shutil.copy("save_models/G_0.pth", f"save_models/{speaker}/G_0.pth")


def main(speaker):
   tasks = [
       resample,
       make_config,
       val,
       preprocess,
       make_model,
   ]

   # リスト内の関数を順番に実行する
   for task in tasks:
      task(speaker)


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--speaker", default="speaker")
   args = parser.parse_args()
   main(args.speaker)
