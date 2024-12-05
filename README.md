# LowResDys

## Requirements
Clone this repo and install dependencies:
```bash
git clone https://github.com/sqrk/LowResDys.git
cd LowResDys
pip install -r requirements.txt
```

## Scripts
### Finetuning
**Whisper**

1. Go to `./whisper/finetune.py`.
2. Change the variables at the beginning of the file if needed, e.g.
```
dataset_name = 'COPAS' #['COPAS', 'easycall', 'torgo', 'uaspeech', 'All', 'All_balanced']
model_name = 'openai/whisper-large-v3'
output_dir = f"./{dataset_name}-whisper-lg-3"
language = 'dutch' #['dutch', 'english', 'italian']
cache_dir = '/l/users/karima.kadaoui/.cache/huggingface'
```
3. Run `python ./whisper/finetune.py`

**MMS**

1. Go to `./mms/finetune.py`.
2. Change the variables at the beginning of the file if needed, e.g.
```
dataset_name = 'COPAS' #[torgo, uaspeech, easycall, COPAS, All, All_balanced]
model_name = 'facebook/mms-1b-all'
output_dir = f"./{dataset_name}-mms1ball"
language = 'ita' #[ita, nld, eng]
cache_dir = '/l/users/karima.kadaoui/.cache/huggingface'
```
3. Run `python ./mms/finetune.py`

### Inference
**Whisper**

1. Go to `./whisper/inference.py`.
2. Change the variables at the beginning of the file if needed, e.g.
```
setting = 'zshot' #[FT, zshot, FTMulti]
dataset_name = 'COPAS' #[torgo, uaspeech, easycall, COPAS, All, All_balanced]
model_name = 'openai/whisper-large-v3'
split = 'test'
language = 'dutch' #['english', 'italian', 'dutch']
cache_dir = '/l/users/karima.kadaoui/.cache/huggingface'
```
3. Run `python ./whisper/inference.py`

**MMS**

1. Go to `./mms/inference.py`.
2. Change the variables at the beginning of the file if needed, e.g.
```
model_name = 'facebook/mms-1b-all'
language = 'eng' #[eng, ita, nld]
dataset_name = "COPAS" #[torgo, uaspeech, easycall, COPAS, All, All_balanced]
setting = 'zshot' #[zshot, FT, FTMulti]
model_name = f'sqrk/{dataset_name}-mms1ball'
split = 'test'
```
3. Run `python ./mms/inference.py`

## Data
The datasets are all uploaded to the following Hugginface dataset `https://huggingface.co/datasets/sqrk/dys_mixture`
You can download a specific dataset using 
```
from datasets import load_dataset

dataset = load_dataset("sqrk/dys_mixture", <dataset_name>)
```
where `dataset_name` can be any of `['COPAS', 'torgo', 'uaspeech', 'easycall', 'All', 'All_balanced']`

The data does **not** need to be downloaded before running the finetuning/inference scripts. The scripts take care of that.

## Checkpoints
| Dataset  | Whisper                                       | MMS                             |
|----------|-----------------------------------------------|---------------------------------|
| COPAS    | [sqrk/COPAS-whisper-lg-3-Nov29](https://huggingface.co/sqrk/COPAS-whisper-lg-3-Nov29)    | [sqrk/COPAS-mms1ball-Nov30](https://huggingface.co/sqrk/COPAS-mms1ball-Nov30)      |
| EasyCall | [sqrk/easycall-whisper-lg-3-Nov29](https://huggingface.co/sqrk/easycall-whisper-lg-3-Nov29)              | [sqrk/easycall-mms1ball-Nov30](https://huggingface.co/sqrk/easycall-mms1ball-Nov30)    |
| TORGO    | [sqrk/torgo-whisper-lg-3-Nov29](https://huggingface.co/sqrk/torgo-whisper-lg-3-Nov29)                 | [sqrk/torgo-mms1ball-Nov30](https://huggingface.co/sqrk/torgo-mms1ball-Nov30)       |
| UASpeech | [sqrk/uaspeech-whisper-lg-3-Nov29](https://huggingface.co/sqrk/uaspeech-whisper-lg-3-Nov29)              | [sqrk/uaspeech-mms1ball-Nov30](https://huggingface.co/sqrk/uaspeech-mms1ball-Nov30)    |
| Multi    | [sqrk/All-lang_tag-whisper-lg-3-Nov30](https://huggingface.co/sqrk/All-lang_tag-whisper-lg-3-Nov30)          | [sqrk/All-mms1ball-Dec1](https://huggingface.co/sqrk/All-mms1ball-Dec1)          |
| Multi_B  | [sqrk/All_balanced-lang_tag-whisper-lg-3-Nov30](https://huggingface.co/sqrk/All_balanced-lang_tag-whisper-lg-3-Nov30) | [sqrk/All_balanced-mms1ball-Dec1](https://huggingface.co/sqrk/All_balanced-mms1ball-Dec1) | 
