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
Go to `./whisper/finetune.py`.
Change the variables at the beginning of the file if needed, e.g.
```
dataset_name = 'COPAS' #['COPAS', 'easycall', 'torgo', 'uaspeech']
model_name = 'openai/whisper-large-v3'
output_dir = f"./{dataset_name}-whisper-lg-3"
language = 'dutch'
cache_dir = '/l/users/karima.kadaoui/.cache/huggingface'
```


## Checkpoints
| Dataset  | Whisper                                       | MMS                             |
|----------|-----------------------------------------------|---------------------------------|
| COPAS    | [sqrk/COPAS-whisper-lg-3-Nov29](https://huggingface.co/sqrk/COPAS-whisper-lg-3-Nov29)    | [sqrk/COPAS-mms1ball-Nov30](https://huggingface.co/sqrk/COPAS-mms1ball-Nov30)      |
| EasyCall | [sqrk/easycall-whisper-lg-3-Nov29](https://huggingface.co/sqrk/easycall-whisper-lg-3-Nov29)              | [sqrk/easycall-mms1ball-Nov30](https://huggingface.co/sqrk/easycall-mms1ball-Nov30)    |
| TORGO    | [sqrk/torgo-whisper-lg-3-Nov29](https://huggingface.co/sqrk/torgo-whisper-lg-3-Nov29)                 | [sqrk/torgo-mms1ball-Nov30](https://huggingface.co/sqrk/torgo-mms1ball-Nov30)       |
| UASpeech | [sqrk/uaspeech-whisper-lg-3-Nov29](https://huggingface.co/sqrk/uaspeech-whisper-lg-3-Nov29)              | [sqrk/uaspeech-mms1ball-Nov30](https://huggingface.co/sqrk/uaspeech-mms1ball-Nov30)    |
| Multi    | [sqrk/All-lang_tag-whisper-lg-3-Nov30](https://huggingface.co/sqrk/All-lang_tag-whisper-lg-3-Nov30)          | [sqrk/All-mms1ball-Dec1](https://huggingface.co/sqrk/All-mms1ball-Dec1)          |
| Multi_B  | [sqrk/All_balanced-lang_tag-whisper-lg-3-Nov30](https://huggingface.co/sqrk/All_balanced-lang_tag-whisper-lg-3-Nov30) | [sqrk/All_balanced-mms1ball-Dec1](https://huggingface.co/sqrk/All_balanced-mms1ball-Dec1) | 
