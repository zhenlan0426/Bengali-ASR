import jax.numpy as jnp
import jax
FLATFORM = 'gcp'
IsTPU = True
dtype = jnp.bfloat16
num_workers = 16
batch_size = 12 * 8
speech_path = 'data/train_mp3s/'
data_path = 'data/train.csv'
output_path = ''
background_audio = 'background/audio/'
background_music = 'background/music/'
model_path = 'model_all'
add_data_path = 'dlsprint/train.csv'
add_audio = 'dlsprint/train_files/'
add_data_path2 = 'text'
add_audio2 = '/mnt/disks/persist/RESPIN/'
add_data_path3 = 'utt_spk_text.tsv'
add_audio3 = '/mnt/disks/persist/SLR/asr_bengali/data/'

#from whisper_jax import FlaxWhisperForConditionalGeneration
from transformers import FlaxWhisperForConditionalGeneration
from functions import *
from functools import partial
import optax
from jax import random
from jax_smi import initialise_tracking
from transformers import AutoTokenizer
import time
import warnings
warnings.filterwarnings('ignore')
pad_to_multiple_of = 64
max_length_gen = 48
epochs = 1
verbose = 10


# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="bn", task="transcribe")
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5")
tokenizer.bos_token = tokenizer.bos_token_id = None
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
text = pd.read_csv(data_path)
add_data = pd.read_csv(add_data_path)
# data source 2
df = pd.read_csv(add_data_path2,names=['code'])
df[['code','sentence']] = df["code"].str.split(" ", n=1, expand=True)
df.sentence = df.sentence.str.strip()
df4 = pd.read_csv(add_data_path3, sep='\t', header=None)
df4.columns = ["audio_path", "__", "sentence"]
df4 = df4.drop("__", axis=1)
"https://github.com/karoldvl/ESC-50/archive/master.zip"
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
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, \
                        collate_fn=partial(collate_fn,tokenizer=tokenizer,feature_extractor=feature_extractor,\
                                           pad_to_multiple_of=pad_to_multiple_of,IsTrain=True,IsTPU=IsTPU,batch_size=batch_size))

with open('/mnt/disks/persist/TXT/cleaned.txt', 'r') as f:
    lines = f.readlines()
train_loader_txt = DataLoader(TxtData(lines), batch_size=32*8, shuffle=True, \
                              collate_fn=partial(collate_fn_txt,tokenizer=tokenizer,\
                                           pad_to_multiple_of=pad_to_multiple_of))

# audio,input_ids,attention_mask = next(iter(train_loader))
# audio.shape,input_ids.shape
# audio,input_ids,attention_mask = jnp.array(audio,dtype=dtype),jnp.array(input_ids),jnp.array(attention_mask)
# load the processor and model
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    model_path, dtype=dtype, _do_init=False)
model.config.forced_decoder_ids = None
model.config.bos_token_id = None
model.config.suppress_tokens = None
model.config.decoder_start_token_id = None
model.generation_config.decoder_start_token_id = [50258, 50302, 50359, 50363]# '<|startoftranscript|><|bn|><|transcribe|><|notimestamps|>
model.generation_config.forced_decoder_ids = None
"""A list of pairs of integers which indicates a mapping from generation indices to token indices 
that will be forced before sampling. For example, [[0, 123]] means the first generated token 
will always be a token of index 123."""
model.generation_config.suppress_tokens = None
model.generation_config.begin_suppress_tokens = None
model.generation_config.bos_token_id = None
# params['lm_head'] = {'kernel':params['model']['decoder']['embed_tokens']['embedding'].T,\
#                      'bias'  :jnp.zeros(model.config.vocab_size,params['model']['decoder']['embed_tokens']['embedding'].dtype)}

from transformers.models.whisper.modeling_flax_whisper import FlaxWhisperDecoder,WhisperConfig
from flax import linen as nn

class DecoderOnly(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.decoder = FlaxWhisperDecoder(self.config,dtype)
        self.lm_head = nn.Dense(self.config.vocab_size,use_bias=True,dtype=dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = False,
    ):
        decoder_out = self.decoder(input_ids=input_ids,\
                                     attention_mask=attention_mask,\
                                     position_ids=(attention_mask.cumsum(-1) * attention_mask) - 1,\
                                     #encoder_hidden_states=encoder_out[0],
                                     deterministic=deterministic
                                    )
        logits = self.lm_head(decoder_out[0])
        return logits

decoder_model = DecoderOnly(model.config,dtype)
decoder_params = {'decoder':params['model']['decoder'],'lm_head':params['lm_head']}

import pickle
# opt_txt = optax.adamw(learning_rate=1.5e-5,
#                   b1=0.9,
#                   b2=0.98,
#                   eps=1e-6,
#                   weight_decay=1e-3,)#1e-1
opt_txt = optax.sgd(learning_rate=2e-5)
opt_txt = optax.chain(optax.clip_by_global_norm(2e-3),opt_txt)
opt_states_txt = opt_txt.init(decoder_params)

opt = optax.adamw(learning_rate=1.5e-5,
                  b1=0.9,
                  b2=0.98,
                  eps=1e-6,
                  weight_decay=1e-3,)#1e-1
opt = optax.chain(optax.clip_by_global_norm(1e-3),opt)
# opt_states = opt.init(params)
filehandler = open(output_path+"opt_states","rb")
opt_states = pickle.load(filehandler)
filehandler.close()

@partial(jax.pmap,axis_name='data')
def train_one_step(params,audio,input_ids,attention_mask,opt_states):
    def loss_fn(params,audio,input_ids,attention_mask):
        out = model(audio,input_ids,decoder_attention_mask=attention_mask,params=params,train=True).logits # (B, L, d)
        return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(out[:,3:-1], input_ids[:,4:])*attention_mask[:,4:])/jnp.sum(attention_mask[:,4:])
    grad_fn = jax.value_and_grad(loss_fn,has_aux=False)
    out = grad_fn(params,audio,input_ids,attention_mask)
    l,grads = jax.lax.pmean(out,'data')
    updates, opt_states = opt.update(grads, opt_states,params=params)
    params = optax.apply_updates(params, updates)
    return params,opt_states,l

@partial(jax.pmap,axis_name='data')
def train_one_step_txt(params,input_ids,attention_mask,opt_states):
    def loss_fn(decoder_params,input_ids,attention_mask):
        logits = decoder_model.apply({'params': decoder_params}, \
                                     input_ids=input_ids,\
                                     attention_mask=attention_mask)
        loss = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits[:,0:-1], input_ids[:,1:])*attention_mask[:,1:])/jnp.sum(attention_mask[:,1:])
        return loss
    grad_fn = jax.value_and_grad(loss_fn,has_aux=False)
    decoder_params = {'decoder':params['model']['decoder'],'lm_head':params['lm_head']}
    out = grad_fn(decoder_params,input_ids,attention_mask)
    l,grads = jax.lax.pmean(out,'data')
    updates, opt_states = opt_txt.update(grads, opt_states)
    decoder_params = optax.apply_updates(decoder_params, updates)
    params['model']['decoder'],params['lm_head'] = decoder_params['decoder'], decoder_params['lm_head']
    return params,opt_states,l

import logging
logging.basicConfig(filename='training.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
def main():
    initialise_tracking()
    IsNaN = False
    devices = jax.devices()
    replicated_params = jax.device_put_replicated(params, devices)
    replicated_opt_states = jax.device_put_replicated(opt_states, devices)
    replicated_opt_states_txt = jax.device_put_replicated(opt_states_txt, devices)

    # start = time.time()
    # train
    train_loss = 0
    train_loss_txt = 0
    skip = skip_txt = 0
    for j,((audio,input_ids,attention_mask),(input_ids_txt,attention_mask_txt)) in enumerate(zip(train_loader,train_loader_txt)):
        # train audio
        if input_ids.shape[1]>=448:
            skip+=1
        else:
            replicated_params,replicated_opt_states,l = train_one_step(replicated_params,\
                                                                    audio,input_ids,attention_mask,replicated_opt_states)
            if jnp.isnan(l[0]).item():
                IsNaN = True
                logging.info(f"audio NaN with shape:{audio.shape,input_ids.shape}")
                raise ValueError("NaN")
            train_loss += l[0].item()
        
        # train txt
        if input_ids_txt.shape[1]>448:
            skip_txt+=1
        else:
            replicated_params,replicated_opt_states_txt,l_txt = train_one_step_txt(replicated_params,\
                                                                input_ids_txt,attention_mask_txt,replicated_opt_states_txt)
            if jnp.isnan(l_txt[0]).item():
                IsNaN = True
                logging.info(f"txt NaN with shape:{input_ids_txt.shape}")
                raise ValueError("NaN")
            train_loss_txt += l_txt[0].item()
        
        if j>0 and j%verbose == 0:
            train_loss /= (verbose-skip)
            train_loss_txt /= (verbose-skip_txt)
            logging.info(f"iterations:{j}, loss: {train_loss:.3f}, loss_txt: {train_loss_txt:.3f}")
            train_loss = 0
            train_loss_txt = 0
            skip = skip_txt = 0
            if j%50 ==0:
                model.save_pretrained(output_path+'model_all',jax.tree_map(lambda x: x[0], replicated_params))
                filehandler = open(output_path+"opt_states","wb")
                pickle.dump(jax.tree_map(lambda x: x[0], replicated_opt_states),filehandler)
                filehandler.close()
                # filehandler = open(output_path+"opt_states_txt","wb")
                # pickle.dump(jax.tree_map(lambda x: x[0], replicated_opt_states_txt),filehandler)
                # filehandler.close()
                
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.info(str(e))
        exit(1)
