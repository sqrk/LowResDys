from datasets import load_dataset, Audio
import numpy as np
import torch, re
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline

model_name = 'facebook/mms-1b-all'
language = 'eng' #[eng, ita, nld]
dataset_name = "COPAS" #[torgo, uaspeech, easycall, COPAS, All, All_balanced]
setting = 'zshot' #[zshot, FT, FTMulti]
model_name = f'sqrk/{dataset_name}-mms1ball'
split = 'test'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DATASET: {dataset_name}\nMODEL: {model_name}\nSETTING: {setting}\nSPLIT: {split}\nLANGUAGE: {language}')

dataset = load_dataset(f'sqrk/dys_mixture', dataset_name, cache_dir='/l/users/karima.kadaoui/.cache/huggingface/datasets')

dataset[split] = dataset[split].select(
    (
        i for i in range(len(dataset[split])) 
        if len(dataset[split][i]['audio']['array']) > 320
    )
)

def filter_inputs(transcription):
    return type(transcription) == str
dataset = dataset.filter(filter_inputs, input_columns=["transcription"])

chars_to_remove_regex = '[,?.!-;:\"\“%\‘\”�\']'
def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
    return batch
dataset[split] = dataset[split].map(remove_special_characters)
print('Removing special chars')

def filter_speakers(speaker_type): #skip control spkrs if any
    return speaker_type == 'dys'

if 'speaker_type' in dataset[split].features:
    print('Filtering out control speakers')
    dataset = dataset.filter(filter_speakers, input_columns=["speaker_type"])

print('Dataset:', dataset[split])

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

target_lang = f'{dataset_name}-{language}'
model = Wav2Vec2ForCTC.from_pretrained(model_name, target_lang=target_lang).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)

processor.tokenizer.set_target_lang(target_lang)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)

count = 0
results = []
output_file_path = f'/l/users/karima.kadaoui/Classes/NLP805/Project/Inference/MMS/{setting}-{dataset_name}-{split}-{language}.tsv'
with open(output_file_path, 'w') as fout:
    fout.write('file_name\treference\tprediction\tspeaker\tdataset\n')


batch_size = 6
num_batches = len(dataset[split]) // batch_size + (1 if len(dataset[split]) % batch_size != 0 else 0)


for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(dataset[split]))
    batch = dataset[split][start_idx:end_idx]
    
    audio_paths = [item['path'] for item in batch['audio']]
    audio_arrays = [np.array(item['array'], dtype=np.float32) for item in batch['audio']]
    speakers = batch['speaker']
    references = batch['transcription']
    datasets = batch['dataset']

    results = pipe(audio_arrays)
    with open(output_file_path, 'a') as fout:
        for i, result in enumerate(results):
            fout.write(f"{audio_paths[i]}\t{references[i]}\t{result['text']}\t{speakers[i]}\t{datasets[i]}\n")
