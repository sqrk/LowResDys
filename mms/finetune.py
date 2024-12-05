import re, time, json, torch, random
start_time = time.time()
from datasets import load_dataset, Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer, TrainerCallback
from dataclasses import dataclass
from typing import Dict, List, Union
from evaluate import load
import numpy as np
from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
import os

dataset_name = 'COPAS' #[torgo, uaspeech, easycall, COPAS, All, All_balanced]
model_name = 'facebook/mms-1b-all'
output_dir = f"./{dataset_name}-mms1ball"
language = 'ita' #[ita, nld, eng]
cache_dir = '/l/users/karima.kadaoui/.cache/huggingface'
print(f'Dataset: {dataset_name}\nModel: {model_name}\nOutput dir: {output_dir}')

class StopAfterTimeLimitCallback(TrainerCallback):
    def __init__(self, max_hours):
        self.max_seconds = max_hours * 3600  

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - start_time > self.max_seconds:
            print("11 hours reached, stopping training.")
            adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
            adapter_file = os.path.join(training_args.output_dir, adapter_file)

            safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})
            trainer.push_to_hub()
            
            control.should_training_stop = True  
        return control
    
class SaveAdapterCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if state.best_metric == state.log_history[-1].get("eval_wer"): #save if best
            adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
            adapter_file = os.path.join(training_args.output_dir, adapter_file)

            safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})
            trainer.push_to_hub()
            print(f"Adapter saved to {adapter_file} with best WER: {state.best_metric}")


dataset = load_dataset("sqrk/dys_mixture", dataset_name, cache_dir=f'{cache_dir}/datasets')

dataset['train'] = dataset['train'].select( #ensure minimum duration (ms) that mms accepts 
    (
        i for i in range(len(dataset['train'])) 
        if len(dataset['train'][i]['audio']['array']) > 320
    )
)

chars_to_remove_regex = '[,?.!-;:\"\“%\‘\”�\']'
def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
    return batch
dataset['train'] = dataset['train'].map(remove_special_characters)


def filter_speakers(speaker_type): #skip control spkrs if any
    return speaker_type == 'dys'

if 'speaker_type' in dataset['train'].features:
    print('Filtering out control speakers')
    dataset = dataset.filter(filter_speakers, input_columns=["speaker_type"])

chars_to_remove_regex = '[,?.!-;:\"\“%\‘\”�\']'
print('Removing special chars')
def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
    return batch

def filter_inputs(transcription):
    return type(transcription) == str
dataset = dataset.filter(filter_inputs, input_columns=["transcription"])

def filter_length(transcription):
    return len(transcription) > 0
dataset = dataset.filter(filter_length, input_columns=["transcription"])

#create vocab for new adapter
all_text = " ".join(dataset['train']["transcription"])
vocab = list(set(all_text))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
target_lang = f'{dataset_name}-{language}'
new_vocab_dict = {target_lang: vocab_dict}

with open(f'./vocab.json', 'w') as vocab_file:
    json.dump(new_vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(f'./', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", target_lang=target_lang)
tokenizer.push_to_hub(output_dir[2:])

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset['train'].train_test_split(test_size=0.1)   #10% held as val

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = processor(text=batch["transcription"]).input_ids
    return batch

train_set = dataset['train']
test_set = dataset['test']

train_set = train_set.map(prepare_dataset, remove_columns=train_set.column_names)
test_set = test_set.map(prepare_dataset, remove_columns=test_set.column_names)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load("wer", cache_dir=f'{cache_dir}/metrics')

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    i1 = random.randrange(0, len(label_str))
    i2 = random.randrange(0, len(label_str))
    print(label_str[i1], pred_str[i1])
    print(label_str[i2], pred_str[i2])
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/mms-1b-all",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
    cache_dir=f'{cache_dir}/models'
)

model.init_adapter_layers()
model.freeze_base_model()

adapter_weights = model._get_adapters()
for param in adapter_weights.values():
    param.requires_grad = True


training_args = TrainingArguments(
  output_dir=output_dir,
  group_by_length=True,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=100,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=200,
  eval_steps=100,
  logging_steps=100,
  learning_rate=1e-3,
  warmup_steps=100,
  save_total_limit=2,
  push_to_hub=True,
  report_to=["wandb"],
  metric_for_best_model="wer",
  greater_is_better=False,
  load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_set,
    eval_dataset=test_set,
    tokenizer=processor.feature_extractor,
    callbacks=[StopAfterTimeLimitCallback(max_hours=11), SaveAdapterCallback(output_dir=training_args.output_dir)]
)

trainer.train()

adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
adapter_file = os.path.join(training_args.output_dir, adapter_file)

safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})
trainer.push_to_hub()
print(f"Final adapter saved to {adapter_file}")