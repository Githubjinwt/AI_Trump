# split speeches into sentences, and clean them

import os
import re
import pandas as pd
import tqdm

def clean(sents: set) -> list:
    new_sents = []
    for sent in list(sents):
        sent = sent.strip('\"').strip(' ')
        if (not '\"' in sent) and (re.findall(r'[a-zA-Z]', sent)):
            new_sents.append(sent)
    return new_sents

in_path = "data/speeches/all_speeches/"
out_path = "data/speeches/speeches_raw.csv"

files = []
file_path = in_path
for filename in os.listdir(file_path):
    with open((os.path.join(file_path, filename)), 'r', encoding='utf-8') as w: files.append(w.read())

sents = set()
for file in tqdm.tqdm(files):
    for sent in file.split('.'):
        sents.add(sent.strip(' '))

sents = clean(sents)
print(len(sents))
speeches = pd.DataFrame(data=sents)
speeches.to_csv(out_path, encoding='utf-8', index=False)
