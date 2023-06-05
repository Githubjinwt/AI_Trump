# parse the collected raw tweets data

import pandas as pd
import json
import re
import tqdm

def Parse(content: str):
    content = content + " "
    # drop the links in the tweets
    for i in re.findall(r'(http.*?\s)', content):
        # content = content.replace(i, '_link_')
        content = content.replace(i, '')
    # pic.tweets.com
    for i in re.findall(r'(pic.twitter.com/.*?)[^a-zA-Z0-9]', content):
        content = content.replace(i, '')

    # drop self quote
    content = content.replace('from Donald Trump: ', '')
    content = content.replace('From Donald Trump: ', '')
    if "Trump" in content:
        return ""

    # drop special punctuations
    content = content.replace('’', '\'')
    content = content.replace('‘', '\'')
    content = content.replace('“', '\'')
    content = content.replace('”', '\'')
    content = content.replace('\"', '\'')
    content = content.replace('...', '')
    for i in ['…','*','&','+','-',':',';','=','<','>','^','–','_','@','#']:
        content = content.replace(i, '')
    content = content.replace('  ', ' ')
    return content.strip('\'').strip(" ").strip('\'')

if __name__ == "__main__":
    in_path = "your @realDonaldTrump tweets csv file"
    out_path = "data/tweets/tweets_raw.csv"
    data = pd.read_csv(in_path).fillna("")
    data.sort_values(by=['date'], inplace=True)

    parsed = []
    for i in tqdm.tqdm(range(len(data))):
        if data.loc[i, 'mentions'] != "" or data.loc[i, 'hashtags'] != "":
            continue
        cnt = Parse(data.loc[i, 'content'])
        if cnt != "":
            parsed.append(cnt)
    dataNew = pd.DataFrame(data=parsed, columns=[['Trump']])
    dataNew.to_csv(out_path, encoding='utf-8', index=False)

    # add classical Trump's quote
    whatdoestrumpthink = "your classical Trump's quote file"
    with open(whatdoestrumpthink, 'r', encoding='utf-8') as f:
        quotes = json.load(f)
    repeats = 2
    for i in range(repeats): # repeat twice
        for quote in quotes['messages']['personalized']:
            parsed.append('He ' + quote)
        for quote in quotes['messages']['non_personalized']:
            parsed.append(quote)
    dataNew = pd.DataFrame(data=parsed, columns=[['Trump']])
    # print(dataNew.tail())
    dataNew.to_csv(out_path, encoding='utf-8', index=False)
