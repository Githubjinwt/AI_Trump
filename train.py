# train your model
import nltk
import numpy as np
import datasets
import pandas as pd
import evaluate
import argparse
from torch import nn
from transformers.utils import logging
from transformers import AutoTokenizer, PreTrainedModel, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers.trainer_utils import set_seed
import warnings
from pathlib import Path
from typing import List, Tuple, Union

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning model.")
    
    parser.add_argument('--batch_size', type = int, default = 32,
                        help = 'input batch size.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--weight_decay', type = float, default = 0.1,
                        help = 'weight decay.')
    parser.add_argument('--epochs', type = int, default = 16,
                        help = 'training epochs.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--model_name', type = str, default = "facebook/bart-large-cnn",
                        help = 'pre-trained model name.')
    parser.add_argument('--model_path', type = str, default = "model/model",
                        help = 'path to save fine-tuned model.')
    parser.add_argument('--checkpoint_path', type = str, default = "model/checkpoint",
                        help = 'path to save checkpoints.')
    parser.add_argument('--data_path', type = str, required=True,
                        help = 'input source data path.')
    
    opt = parser.parse_args()

    return opt

opt = parse_option()

# initialize parameters
max_input_length = 1024 # input, source text
max_target_length = 128 # summary, target text
sample_rate = 0.8 # 0.8 of data for train, 0.2 of data for test
batch_size = opt.batch_size
epoch = opt.epochs
lr = opt.learning_rate
weight_decay = opt.weight_decay
model_path = opt.model_path
checkpoint_path = opt.checkpoint_path
set_seed(opt.seed)
model_name = opt.model_name
data_path = opt.data_path

nltk.download('punkt')

# load and process dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def preprocess_function(examples):
    inputs = [doc for doc in examples["Neutral"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    # with tokenizer.as_target_tokenizer():
    labels = tokenizer(text_target=examples["Trump"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

raw_data = pd.read_csv(data_path, encoding='utf-8')
raw_data = raw_data[['Trump', 'Neutral']]
raw_data = raw_data.sample(frac=1).reset_index(drop=True)
sep = int(sample_rate * len(raw_data))

train_dataset = datasets.Dataset.from_dict(raw_data[:sep])
test_dataset = datasets.Dataset.from_dict(raw_data[sep:])
data = datasets.DatasetDict({'train':train_dataset, 'test':test_dataset})

# metric = load_metric("rouge")
metric = evaluate.load("rouge")

tokenized_datasets = data.map(preprocess_function, batched=True)
# print(tokenized_datasets)

from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to(device)


# model prepared, start copying
logger = logging.get_logger(__name__)

def copy_layers(src_layers: nn.ModuleList, dest_layers: nn.ModuleList, layers_to_copy: List[int]) -> None:
    layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
    assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())


# copy layers, BART's encoder and decoder have both 12 layers
LAYERS_TO_COPY = {
    # maps  num layers in teacher -> num_layers in student -> which teacher layers to copy.
    12: {
        1: [0],  # This says that if the teacher has 12 layers and the student has 1, copy layer 0 of the teacher
        2: [0, 6],
        3: [0, 6, 11],
        4: [0, 4, 8, 11],
        6: [0, 2, 4, 7, 9, 11],
        9: [0, 1, 2, 4, 5, 7, 9, 10, 11],
        12: list(range(12)),
    },
}
LAYERS_TO_SUPERVISE = {
    # maps  num layers in student -> which teacher layers to copy.
    12: {1: [11], 2: [5, 11], 3: [3, 7, 11], 6: [1, 3, 5, 8, 10, 11]},
}


# prepare student model from teacher model

def pick_layers_to_copy(n_student, n_teacher):
    try:
        val = LAYERS_TO_COPY[n_teacher][n_student]
        return val
    except KeyError:
        if n_student != n_teacher:
            warnings.warn(
                f"no hardcoded layers to copy for teacher {n_teacher} -> student {n_student}, defaulting to first {n_student}"
            )
        return list(range(n_student))

def create_student_by_copying_alternating_layers(
    teacher: Union[str, PreTrainedModel],
    save_path: Union[str, Path] = "student",
    e: Union[int, None] = None,
    d: Union[int, None] = None,
    e_layers_to_copy=None,
    d_layers_to_copy=None,
    **extra_config_kwargs
) -> Tuple[PreTrainedModel, List[int], List[int]]:
    if isinstance(teacher, str): # string: load model
        AutoTokenizer.from_pretrained(teacher).save_pretrained(save_path)
        teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher).eval()
    else:
        assert isinstance(teacher, PreTrainedModel), f"teacher must be a model or string got type {type(teacher)}"
    init_kwargs = teacher.config.to_diff_dict()

    teacher_e, teacher_d = teacher.config.encoder_layers, teacher.config.decoder_layers
    if e is None:
        e = teacher_e
    if d is None:
        d = teacher_d
    init_kwargs.update({"encoder_layers": e, "decoder_layers": d})

    # Kwargs to instantiate student: teacher kwargs with updated layer numbers + **extra_config_kwargs
    init_kwargs.update(extra_config_kwargs)

    # Copy weights
    student_cfg = teacher.config_class(**init_kwargs)
    student = AutoModelForSeq2SeqLM.from_config(student_cfg)
    # Start by copying the full teacher state dict this will copy the first N teacher layers to the student.
    info = student.load_state_dict(teacher.state_dict(), strict=False)
    assert info.missing_keys == [], info.missing_keys  # every student key should have a teacher keys.

    # Decide which layers of the teacher to copy. Not exactly alternating -- we try to keep first and last layer.
    if e_layers_to_copy is None:
        e_layers_to_copy: List[int] = pick_layers_to_copy(e, teacher_e)
    if d_layers_to_copy is None:
        d_layers_to_copy: List[int] = pick_layers_to_copy(d, teacher_d)

    copy_layers(teacher.model.encoder.layers, student.model.encoder.layers, e_layers_to_copy)
    copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, d_layers_to_copy)

    student.config.init_metadata = dict(
        teacher_type=teacher.config.model_type,
        copied_encoder_layers=e_layers_to_copy,
        copied_decoder_layers=d_layers_to_copy,
    )

    return student.to(device)

model = create_student_by_copying_alternating_layers(model, model_path, 12, 9)


# train student model

args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_path,
    num_train_epochs=epoch,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,
    weight_decay=weight_decay,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=200,
    save_total_limit=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key:value for key, value in result.items()}
    
    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained(model_path)
