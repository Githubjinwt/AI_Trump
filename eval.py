import pandas as pd
import numpy as np
import tqdm
import torch
import gc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from evaluation.similarity import calc_bleu, calc_semantic_similarity, calc_lexical_accuracy
from evaluation.fluency import calc_fluency

def transfer(Input, tokenizer, model):
    input_ids = tokenizer.batch_encode_plus([Input], max_length=1024, return_tensors='pt', truncation=True)['input_ids']
    output_ids = model.generate(input_ids, num_beams=1, length_penalty=2, max_length=100, min_length=5, no_repeat_ngram_size=3)
    output_txt = tokenizer.decode(output_ids.squeeze(), skip_special_tokens=True)
    return output_txt

def generate_predict(data_path, pred_path, model_path, tok_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    data = pd.read_csv(data_path, encoding='utf-8')
    for i in tqdm.tqdm(range(len(data))):
        data.loc[i, 'Predict'] = transfer(data.loc[i, 'Neutral'], tokenizer, model)
        data.loc[i, 'Reference'] = data.loc[i, 'Trump']
    data = data[['Predict', 'Reference']]
    data.to_csv(pred_path, encoding='utf-8')

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def count_score(input_path, pred_path):
    res = pd.read_csv(pred_path, encoding='utf-8')
    Predict = list(res['Predict'])
    Reference = list(res['Reference'])

    res = pd.read_csv(input_path, encoding='utf-8')
    Input = list(res['Neutral'])

    # similarity
    BLEU = calc_bleu(Reference, Predict)

    lexical_accu = calc_lexical_accuracy(Input, Reference, Predict)

    semantic_sim_stats = calc_semantic_similarity(Reference, Predict)
    semantic_sim = semantic_sim_stats.mean()
    cleanup()

    # fluency, lower is better
    FL_input = calc_fluency(Reference)
    FL_pred = calc_fluency(Predict)

    return BLEU, lexical_accu, semantic_sim, FL_input, FL_pred

if __name__ == "__main__":
    generate_predict(data_path="your data path",
                     pred_path="your predict path",
                     model_path="your model path",
                     tok_path="your checkpoint path")
    BLEU, lexical_accu, semantic_sim, FL_input, FL_pred = count_score(
        input_path="your data path",
        pred_path="your predict path")
