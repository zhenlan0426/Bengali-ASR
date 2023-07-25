from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers import WhisperFeatureExtractor
import librosa
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import jax
import jax.numpy as jnp

class AudioDataset(Dataset):
    def __init__(self, text,speech_path):
        self.text = text
        self.speech_path = speech_path
    def __len__(self):
        return self.text.shape[0]
    def __getitem__(self,idx):
        audio = librosa.load(self.speech_path+self.text.id.iloc[idx]+'.mp3')[0]
        audio = librosa.resample(audio, orig_sr=32000, target_sr=16000)
        txt = self.text.sentence.iloc[idx]
        return audio,txt
    
def get_len(audio_list,pad_to_multiple_of):
    # pad_to_multiple_of is applied in feature_extractor before FFT. But we want the output of FFT to be a multiple of.
    # 160 is waveform length -> specgram length
    max_len = max(a.shape[0] for a in audio_list)
    return (max_len//160//pad_to_multiple_of+1)*pad_to_multiple_of*160

def collate_fn(data,feature_extractor,tokenizer,pad_to_multiple_of):
    # data: is a list of tuples with [(audio:1d Array,txt:List of text),...]
    audio,txt = zip(*data)
    audio = feature_extractor(audio,sampling_rate=16000,do_normalize=True,\
                              max_length=get_len(audio,pad_to_multiple_of),return_tensors='np',\
                              return_attention_mask=False)['input_features']
    txt = tokenizer.batch_encode_plus(txt,padding=True,return_attention_mask=True,pad_to_multiple_of=pad_to_multiple_of,return_tensors='np')
    return audio, txt['input_ids'], txt['attention_mask']