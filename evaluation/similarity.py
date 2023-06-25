import torch
import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from flair.data import Sentence
from flair.embeddings import FlairEmbeddings
from torch.nn.functional import cosine_similarity

from semantic_text_similarity.models import WebBertSimilarity
# from semantic_text_similarity.models import ClinicalBertSimilarity

def calc_bleu(inputs, preds):
    bleu_sim = 0
    counter = 0
    print('Calculating BLEU similarity')
    for i in range(len(inputs)):
        if len(inputs[i]) > 3 and len(preds[i]) > 3:
            bleu_sim += sentence_bleu([inputs[i]], preds[i])
            counter += 1

    return float(bleu_sim / counter)


def flair_sim(inputs, preds):
    print('Calculating flair embeddings similarity')
    batch_size = 32
    inp_embed = []
    pred_embed = []

    embedder = FlairEmbeddings('news-forward')

    for i in range(0, len(inputs), batch_size):
        inp_part = [Sentence(sent) for sent in inputs[i:i + batch_size]]
        pred_part = [Sentence(sent) for sent in preds[i:i + batch_size]]

        inp_part = embedder.embed(inp_part)
        pred_part = embedder.embed(pred_part)

        for j in range(batch_size):
            if ((i + j) < len(inputs)):
                inp_sent_vec = torch.zeros(2048).cuda()
                pred_sent_vec = torch.zeros(2048).cuda()

                for k in range(len(inp_part[j])):
                    inp_sent_vec += inp_part[j][k].embedding
                inp_embed.append(inp_sent_vec.cpu() / (k + 1))

                for k in range(len(pred_part[j])):
                    pred_sent_vec += pred_part[j][k].embedding
                pred_embed.append(pred_sent_vec.cpu() / (k + 1))

    emb_sim = cosine_similarity(torch.stack(inp_embed), torch.stack(pred_embed))

    return emb_sim


def calc_lexical_accuracy(inputs, preds, refes):
    assert len(inputs) == len(preds)
    assert len(inputs) == len(refes)

    print('Calculating flair embeddings similarity')
    batch_size = 32
    inp_embed = []
    pred_embed = []
    ref_embed = []

    embedder = FlairEmbeddings('news-forward')

    for i in tqdm.tqdm(range(0, len(inputs), batch_size)):
        inp_part = [Sentence(sent) for sent in inputs[i:i + batch_size]]
        pred_part = [Sentence(sent) for sent in preds[i:i + batch_size]]
        ref_part = [Sentence(sent) for sent in refes[i:i + batch_size]]

        inp_part = embedder.embed(inp_part)
        pred_part = embedder.embed(pred_part)
        ref_part = embedder.embed(ref_part)

        for j in range(batch_size):
            if ((i + j) < len(inputs)):
                inp_sent_vec = torch.zeros(2048).cuda()
                pred_sent_vec = torch.zeros(2048).cuda()
                ref_sent_vec = torch.zeros(2048).cuda()

                for k in range(len(inp_part[j])):
                    inp_sent_vec += inp_part[j][k].embedding
                inp_embed.append(inp_sent_vec.cpu() / (k + 1))

                for k in range(len(pred_part[j])):
                    pred_sent_vec += pred_part[j][k].embedding
                pred_embed.append(pred_sent_vec.cpu() / (k + 1))

                for k in range(len(ref_part[j])):
                    ref_sent_vec += ref_part[j][k].embedding
                ref_embed.append(ref_sent_vec.cpu() / (k + 1))

    accu = 0
    for i in tqdm.tqdm(range(len(inputs))):
        emb_sim_inp = cosine_similarity(torch.stack([inp_embed[i]]), torch.stack([pred_embed[i]]))
        emb_sim_ref = cosine_similarity(torch.stack([ref_embed[i]]), torch.stack([pred_embed[i]]))
        if emb_sim_inp < emb_sim_ref:
            accu += 1

    return accu / len(inputs)


def calc_semantic_similarity(inputs, preds):
    assert len(inputs) == len(preds)

    print('Calculating semantic similarity')
    web_model = WebBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction
    # clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction
    scores = web_model.predict(list(zip(inputs, preds)))
    return np.array(scores)
