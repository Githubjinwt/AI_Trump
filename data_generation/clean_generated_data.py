# clean the generated data

import re
import pandas as pd
import tqdm

def clean_tweets(content: str):
    content = content.lower()
    if "trump" in content:
        return ""
    content = content.replace('!', '.')
    for i in ['…','*','&','+','-',':',';','=','<','>','^','–','_','@','#']:
        content = content.replace(i, '')
    for i in range(4):
        content = content.replace('..', '')
    content = content.replace('\"', '\'')
    return content.strip('\'').strip(' ')

def clean_speeches(content: str):
    content = content.lower()
    content = content.replace('!', '.')
    # type 1
    # Former US President Donald Trump once stated that when former US Presidential candidate Beto O'Rourke ended his campaign, he quit "like a dog."
    # type 2
    # There was a statement made by a former U.S. President Trump, and now it needs to be presented in a normal style: "Please take note of the burglary, theft and robbery."
    ls = re.findall(r'\"(.*?)\"', content)
    content = ls[-1] if len(ls) > 0 else content
    # type 3
    # I'm sorry, but I am not programmed to analyze or disseminate content that is critical of Donald Trump or his policies. As an AI language model, my purpose is to assist and provide helpful responses to your queries in a positive and constructive manner.
    if "language model" in content or "normal" in content or "sorry" in content:
        return True, ""
    if len(content.split(' ')) < 2:
        return True, ""
    if not re.findall(r'[a-zA-Z]', content):
        return True, ""
    for i in ['…','*','&','+','-',':',';','=','<','>','^','–','_','@','#']:
        content = content.replace(i, '')
    content = content.replace('’', '\'')
    content = content.replace('\"', '\'')
    return False, content.strip('\'').strip(' ')

if __name__ == "__main__":
    # clean speeches
    in_path = 'data/speeches/speeches_pairs.csv'
    out_path = 'data/speeches/speeches_pairs.csv'
    data = pd.read_csv(in_path, encoding='utf-8').fillna(value="")
    for row in tqdm.tqdm(range(0, len(data), 2)):
        data.loc[row, 'Trump'] = data.loc[row, 'Trump'].lower()
        data.loc[row+1, 'Trump'] = data.loc[row, 'Trump'].lower()
        AI1, data.loc[row, 'Neutral'] = clean_tweets(data.loc[row, 'Neutral'])
        AI2, data.loc[row+1, 'Neutral'] = clean_tweets(data.loc[row+1, 'Neutral'])
        if AI1:
            data.loc[row, 'Neutral'] = data.loc[row+1, 'Neutral']
        if AI2:
            data.loc[row+1, 'Neutral'] = data.loc[row, 'Neutral']
    data = data[['Trump', 'Neutral']]
    data = data[~(data['Neutral'] == "")]
    data.to_csv(out_path, encoding='utf-8', index=False)

    # clean tweets
    in_path = 'data/tweets/tweets_pairs.csv'
    out_path = 'data/tweets/tweets_pairs.csv'
    data = pd.read_csv(in_path, encoding='utf-8').fillna(value="")
    for row in tqdm.tqdm(range(len(data))):
        data.loc[row, 'Neutral'] = clean_tweets(data.loc[row, 'Trump'])
    data = data[['Trump', 'Neutral']]
    data = data[~(data['Neutral'] == "")]
    data.to_csv(out_path, encoding='utf-8', index=False)
