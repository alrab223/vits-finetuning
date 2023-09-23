## Making TTS Easy
This repository is intended to make it easier to create speech synthesis using VITS.

## How to use 


### Install

```sh
git clone https://github.com/alrab223/vits-finetuning_auto
```


```sh
pip install -r requirements.txt
```

## Download pre-trained model

Put these pre-trained models in the "save_models" 
- [G_0-p.pth](https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/G_0-p.pth)

- [D_0-p.pth](https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/D_0-p.pth)

## Create datasets
Put the set of audio files (wav) you want to train into the "dataset" folder.


- About 50 audio-text pairs will suffice and 100-600 epochs could have quite good performance, but more data may be better.

- Audio files should be >=1s and <=10s.

Run the following script
```sh
python transcribe.py --speaker XXX
```

For a single speaker, the following text file is generated.

Correct any incorrectly transcribed text.

```sh
path/to/XXX.wav|transcript
```

The following command creates a dataset.

```sh
python make.py --speaker XXX
```

## Train

```sh
python train.py -c <CONFIG> -m <XXX>
```