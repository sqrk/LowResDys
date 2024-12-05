from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset, Audio
import numpy as np
import torch

setting = 'zshot' #[FT, zshot, FTMulti]
dataset_name = 'COPAS' #[torgo, uaspeech, easycall, COPAS, All, All_balanced]
model_name = 'openai/whisper-large-v3'
split = 'test'
language = 'dutch' #['english', 'italian', 'dutch']
cache_dir = '/l/users/karima.kadaoui/.cache/huggingface'

print(f'DATASET: {dataset_name}\nMODEL: {model_name}\nSETTING: {setting}\nSPLIT: {split}\nLANGUAGE: {language}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset(f'sqrk/dys_mixture', dataset_name, cache_dir=f'{cache_dir}/datasets')

def filter_length(transcription): #ensuring no empty refs
    return len(transcription) > 0
dataset = dataset.filter(filter_length, input_columns=["transcription"])
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


print('dataset:', dataset)
print('1st elem:', dataset['test'][0])

processor = AutoProcessor.from_pretrained(model_name, cache_dir=f'{cache_dir}/models')
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, cache_dir=f'{cache_dir}/models')

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)

count = 0
results = []
output_file_path = f'./{setting}-{dataset_name}-{split}-{language}.tsv'
with open(output_file_path, 'w') as fout:
    fout.write('file_name\treference\tprediction\tspeaker\tdataset\n')

batch_size = 8
num_batches = len(dataset[split]) // batch_size + (1 if len(dataset[split]) % batch_size != 0 else 0)

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(dataset[split]))
    batch = dataset[split][start_idx:end_idx]

    audio_paths = [item['path'] for item in batch['audio']]
    speakers = batch['speaker']
    references = batch['transcription']
    datasets = batch['dataset']
    audio_arrays = [np.array(item['array'], dtype=np.float32) for item in batch['audio']]
    
    results = pipe(audio_arrays, generate_kwargs={"language": language})
    with open(output_file_path, 'a') as fout:    
        for i, result in enumerate(results):
            fout.write(f"{audio_paths[i]}\t{references[i]}\t{result['text']}\t{speakers[i]}\t{datasets[i]}\n")

