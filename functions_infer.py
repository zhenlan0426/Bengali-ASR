import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

# this is specific to whisper model
pad_token_id = eos_token_id = 50257
bos_ids = np.array([[50258, 50302, 50359, 50363]]) # '<|startoftranscript|><|bn|><|transcribe|><|notimestamps|>
mask_ids = np.ones((1,4),dtype=np.int64)
# data sr and required sr for model
orig_sr=32000; target_sr=16000

class Inference(Dataset):
    def __init__(self, speech_path, df):
        self.df = df
        self.speech_path = speech_path
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self,idx):
        audio = librosa.load(self.speech_path + self.df.id.iloc[idx]+'.mp3',sr=target_sr)[0]
        return audio

def collate_fn_infer(data,feature_extractor):
    # data: is a list of tuples with [(audio:1d Array,txt:List of text),...]
    audio = feature_extractor(data,sampling_rate=16000,do_normalize=True,\
                              max_length=max(a.shape[0] for a in data),return_tensors='np',\
                              return_attention_mask=False)['input_features']
    return audio