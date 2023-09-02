import torch
IsTPU = False
# batch_size = 4
device = 'cuda'
dtype = torch.float32
speech_path = 'data/train_mp3s/'
data_path = 'data/train.csv'
num_workers = 16
accumulation_steps = 16
clip = 8e-4
lr = 8e-6
output_path = ''
model_path = '/home/zhenlan/Desktop/Projects/Bengali ASR/best_wav2vec/best.pth'
background_audio = "/home/zhenlan/Desktop/Projects/Bengali ASR/ESC-50-master/audio"
background_music = "/home/zhenlan/Desktop/Projects/Bengali ASR/ESC-50-master/music"
add_data_path = 'dlsprint/train.csv'
add_audio = 'dlsprint/train_files/'
add_data_path2 = 'text_train'
add_audio2 = 'RESPIN/'
add_data_path3 = '/home/zhenlan/Desktop/Projects/Bengali ASR/asr_bengali/utt_spk_text.tsv'
add_audio3 = '/home/zhenlan/Desktop/Projects/Bengali ASR/asr_bengali/data/'
#from whisper_jax import FlaxWhisperForConditionalGeneration
from functions import *
import torch.nn as nn
from functools import partial
from transformers.modeling_outputs import BaseModelOutput
from transformers import WhisperForConditionalGeneration
from transformers.generation.configuration_utils import GenerationConfig
from transformers import AutoTokenizer,Wav2Vec2Processor
from torch.optim import AdamW,Adam,SGD
from torch.nn.utils import clip_grad_value_
from torch.cuda.amp import GradScaler
import time
import math
import pickle
import warnings
warnings.filterwarnings('ignore')
model = torch.load(model_path).to(device)
paras = model.parameters()
opt = AdamW(paras,lr = lr,amsgrad=True,weight_decay=6e-4)
#opt = Adam(paras,lr = lr)
# opt = SGD(paras,lr = lr)
epochs = 1
verbose = 250
processor = Wav2Vec2Processor.from_pretrained("/home/zhenlan/Desktop/Projects/Bengali ASR/best_wav2vec")
tokenizer = processor.tokenizer
feature_extractor = processor.feature_extractor
text = pd.read_csv(data_path)
add_data = pd.read_csv(add_data_path)
# data source 2
df = pd.read_csv(add_data_path2,names=['code'])
df[['code','sentence']] = df["code"].str.split(" ", n=1, expand=True)
df.sentence = df.sentence.str.strip()
df4 = pd.read_csv(add_data_path3, sep='\t', header=None)
df4.columns = ["audio_path", "__", "sentence"]
df4 = df4.drop("__", axis=1)
# "https://github.com/karoldvl/ESC-50/archive/master.zip"
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    Compose,
    Gain,
    GainTransition,
    OneOf,
    PitchShift,
    PolarityInversion,
    TimeStretch,
    OneOf,
    )

# define augmentation
augmentation = Compose(
    [
        TimeStretch(min_rate=0.9, max_rate=1.1, p=1, leave_length_unchanged=False),
        Gain(min_gain_db=-6, max_gain_db=6, p=1),
        GainTransition(min_gain_db=-6, max_gain_db=6),
        PitchShift(min_semitones=-4, max_semitones=4, p=1),
        OneOf([AddBackgroundNoise(sounds_path=background_audio, min_snr_in_db=1.0, max_snr_in_db=5.0, noise_transform=PolarityInversion(), p=1.0),\
              AddBackgroundNoise(sounds_path=background_music, min_snr_in_db=1.0, max_snr_in_db=5.0, noise_transform=PolarityInversion(), p=1.0)]),
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1),
    ]
)

with open("lengths", "rb") as fp:
    lengths = pickle.load(fp)
dataset1 = AudioDataset(text,speech_path,lambda x:x.id+'.mp3',\
                       augmentation,orig_sr=32000, target_sr=16000)
dataset2 = AudioDataset(add_data,add_audio,\
                        lambda x:x.path,augmentation,orig_sr=48000, target_sr=16000)
dataset3 = AudioDataset(df,add_audio2,\
                        lambda x:x.code.split('_')[-1] + '.wav',augmentation,orig_sr=16000, target_sr=16000)
dataset4 = AudioDataset(df4,add_audio3,\
                        lambda x:x.audio_path + '.flac',augmentation,orig_sr=16000, target_sr=16000)
                    
dataset = Add2Data(dataset1,dataset2)
dataset = Add2Data(dataset,dataset3)
dataset = Add2Data(dataset,dataset4)
train_loader = DataLoader(dataset, num_workers=num_workers, \
                            collate_fn=partial(collate_fn_pt_wav2vec,tokenizer=tokenizer,feature_extractor=feature_extractor),\
                            batch_sampler=DynamicBucketingBatchSampler(lengths))
#%debug
# audio,input_ids,attention_mask = next(iter(train_loader))
# audio,input_ids,attention_mask = torch.asarray(audio,dtype=torch.float32,device=device),torch.asarray(input_ids,dtype=torch.long,device=device),torch.asarray(attention_mask,dtype=torch.float32,device=device)
# audio.shape,input_ids.shape
# attention_mask.sum()/attention_mask.shape[0]/attention_mask.shape[1]
# audio,input_ids,attention_mask = jnp.array(audio,dtype=dtype),jnp.array(input_ids),jnp.array(attention_mask)
# lengths1 = [librosa.load(speech_path + text.iloc[i].id+'.mp3')[0].shape[0] for i in range(text.shape[0])]
# lengths2 = [librosa.load(add_audio + add_data.iloc[i].path)[0].shape[0] for i in range(add_data.shape[0])]
# lengths3 = [librosa.load(add_audio2 + df.iloc[i].code.split('_')[-1] + '.wav')[0].shape[0] for i in range(df.shape[0])]
# import matplotlib.pyplot as plt
# plt.hist(lengths1,range=(0,535815))
# plt.hist(lengths3,range=(0,535815))
# plt.hist(lengths2,range=(0,535815))
# import pickle
# lengths = lengths1 + lengths2 + lengths3
# with open("lengths", "wb") as fp:
#     pickle.dump(lengths, fp)
import logging
logging.basicConfig(filename='training_best.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
def main():
    use_amp = True
    model.train()
    np.random.seed()
    train_loss = 0
    skip = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # Training #
    for j,(audio,input_ids,attention_mask) in enumerate(train_loader):
        audio,input_ids,attention_mask = torch.asarray(audio,device=device),\
                                        torch.asarray(input_ids,device=device),\
                                        torch.asarray(attention_mask,device=device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            loss = model(audio,attention_mask=attention_mask,labels=input_ids).loss
        if math.isinf(loss.item()) or math.isnan(loss.item()):
            skip += 1
            continue
        train_loss += loss.item()
        scaler.scale(loss).backward()
        if j%accumulation_steps == 0:
            scaler.unscale_(opt)
            clip_grad_value_(paras,clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        #print(torch.any(torch.isnan(logits)),scaler.get_scale())
        if j>0 and j%verbose == 0:
            train_loss /= (verbose-skip)
            logging.info(f"iterations:{j}, loss: {train_loss:.3f}")
            train_loss = 0
            skip = 0
        if j%1000==0:
            torch.save(model, model_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.info(str(e))
        exit(1)

