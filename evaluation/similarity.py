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


def calc_semantic_similarity(inputs, preds):
    assert len(inputs) == len(preds)

    web_model = WebBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction
    # clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction
    scores = web_model.predict(list(zip(inputs, preds)))
    return np.array(scores)
