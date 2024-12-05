import time, random 
start_time = time.time()
import torch, re, evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoProcessor, AutoModelForSpeechSeq2Seq, TrainerCallback
import pyarrow.dataset as ds

dataset_name = 'COPAS' #['COPAS', 'easycall', 'torgo', 'uaspeech', 'All', 'All_balanced']
model_name = 'openai/whisper-large-v3'
output_dir = f"./{dataset_name}-whisper-lg-3"
language = 'dutch' #['dutch', 'english', 'italian']
cache_dir = '/l/users/karima.kadaoui/.cache/huggingface'
print(f'Dataset: {dataset_name}\nModel: {model_name}\nOutput dir: {output_dir}')

processor = AutoProcessor.from_pretrained(model_name, task="transcribe", cache_dir=f'{cache_dir}/models') #TODO Change whisper
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0] #todo mms change features to values
    processor.tokenizer.set_prefix_tokens(language=language, task="transcribe") 
    batch["labels"] = processor.tokenizer(batch['transcription']).input_ids
    return batch

dataset = load_dataset("sqrk/dys_mixture", dataset_name, cache_dir=f'{cache_dir}/datasets')

#Pre-processing
def filter_inputs(transcription):
    return type(transcription) == str
dataset = dataset.filter(filter_inputs, input_columns=["transcription"])

def filter_speakers(speaker_type):
    return speaker_type == 'dys' 
if 'speaker_type' in dataset['train'].features: #skip control spkrs if any
    print('Filtering out control speakers')
    dataset = dataset.filter(filter_speakers, input_columns=["speaker_type"])

chars_to_remove_regex = '[,?.!-;:\"\“%\‘\”�\']'
print('Removing special chars')
def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
    return batch

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset['train'].train_test_split(test_size=0.1) #10% held as val
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, cache_dir=f'{cache_dir}/models') 
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

class StopAfterTimeLimitCallback(TrainerCallback):
    def __init__(self, max_hours):
        self.max_seconds = max_hours * 3600 

    def on_step_end(self, args, state, control, **kwargs):
        #stop once 11h is reached to save tokenizer correctly
        if time.time() - start_time > self.max_seconds:
            print("11 hours reached, stopping training n saving")
            trainer.save_model()
            processor.tokenizer.save_pretrained(training_args.output_dir)
            trainer.push_to_hub()
            
            control.should_training_stop = True 
        return control

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = evaluate.load("wer", cache_dir=f'{cache_dir}/metrics')

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    i = random.randrange(0, len(pred_str) - 1)
    print(label_str[i], '\t', pred_str[i])
    print(label_str[i+1], '\t', pred_str[i+1])
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir, 
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=100,
    num_train_epochs=100,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=150,
    save_steps=200,
    eval_steps=100,
    logging_steps=100,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    save_total_limit=2
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[StopAfterTimeLimitCallback(max_hours=11)]
)

# trainer.train(resume_from_checkpoint=True)
trainer.train()

trainer.save_model()
processor.tokenizer.save_pretrained(training_args.output_dir)
trainer.push_to_hub()
