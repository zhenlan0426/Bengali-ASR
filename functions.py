from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers import WhisperFeatureExtractor
import librosa
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import jax
import jax.numpy as jnp
# this is specific to whisper model
pad_token_id = eos_token_id = 50257
bos_ids = np.array([[50258, 50302, 50359, 50363]]) # '<|startoftranscript|><|bn|><|transcribe|><|notimestamps|>
mask_ids = np.ones((1,4),dtype=np.int64)

class AudioDataset(Dataset):
    def __init__(self, text,speech_path):
        self.text = text
        self.speech_path = speech_path
    def __len__(self):
        return self.text.shape[0]
    def __getitem__(self,idx):
        audio = librosa.load(self.speech_path+self.text.id.iloc[idx]+'.mp3')[0]
        audio = librosa.resample(audio, orig_sr=32000, target_sr=16000)
        txt = self.text.sentence.iloc[idx][:-1] # remove "|"
        return audio,txt
    
def get_len(audio_list,pad_to_multiple_of):
    # pad_to_multiple_of is applied in feature_extractor before FFT. But we want the output of FFT to be a multiple of.
    # 160 is waveform length -> specgram length
    max_len = max(a.shape[0] for a in audio_list)
    return (max_len//160//pad_to_multiple_of+1)*pad_to_multiple_of*160

def collate_fn(data,feature_extractor,tokenizer,pad_to_multiple_of,IsTrain):
    # data: is a list of tuples with [(audio:1d Array,txt:List of text),...]
    audio,txt = zip(*data)
    audio = feature_extractor(audio,sampling_rate=16000,do_normalize=True,\
                              max_length=get_len(audio,pad_to_multiple_of),return_tensors='np',\
                              return_attention_mask=False)['input_features']
    if IsTrain:
        txt = tokenizer.batch_encode_plus(txt,padding=True,return_attention_mask=True,pad_to_multiple_of=pad_to_multiple_of,return_tensors='np')
        input_ids,attention_mask = txt['input_ids'], txt['attention_mask']
        # pad/eos is based on tokenizer but not consistent with model
        pad_idx = np.where(input_ids==tokenizer.pad_token_id)
        eos_idx = np.where(input_ids==tokenizer.eos_token_id)
        input_ids[pad_idx] = pad_token_id
        input_ids[eos_idx] = eos_token_id
        input_ids = np.concatenate([np.broadcast_to(bos_ids,(input_ids.shape[0],4)),input_ids],1)
        attention_mask = np.concatenate([np.broadcast_to(mask_ids,(input_ids.shape[0],4)),attention_mask],1)
        return audio, input_ids,attention_mask # mask (n,l-4) -4 bc it does not have the special token at the begining
    else:
        return audio,txt
