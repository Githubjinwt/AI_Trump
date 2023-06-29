# two methods for NLP data augmentation: back translatation & character substitution

import http.client
import hashlib
import json
import urllib
import random
import pandas as pd
import tqdm
# source: https://github.com/makcedward/nlpaug.git
import nlpaug.augmenter.char as nac

appid = "your baidu translator appid"
secretKey = "your baidu translator secret key"

def baidu_translate(content, fromLang="zh", toLang="en"):
    httpClient = None
    myurl = '/api/trans/vip/translate'
    q = content
    salt = random.randint(32768, 65536)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        response = httpClient.getresponse()
        jsonResponse = response.read().decode("utf-8")
        js = json.loads(jsonResponse)
        dst = str(js["trans_result"][0]["dst"])
        return dst
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()

def back_translate(text):
    zh = baidu_translate(text, fromLang='en', toLang='zh')
    en = baidu_translate(zh, fromLang='zh', toLang='en')
    return en

def substitute_by_keyboard(text):
    aug = nac.KeyboardAug(aug_char_min=1, aug_char_max=1, aug_word_min=1, aug_word_max=2)
    augmented_text = aug.augment(text)
    return augmented_text

if __name__=='__main__':    
    in_path = "your cleaned data"
    out_path = "your augmented data"
    dataOld = pd.read_csv(in_path, encoding='utf-8')
    dataNew = pd.DataFrame()

    for i in tqdm.tqdm(range(len(dataOld))):
        dataNew.loc[i, 'Trump'] = dataOld.loc[i, 'Trump']
        dataNew.loc[i, 'Neutral'] = back_translate(dataOld.loc[i, 'Neutral'])
        dataNew.loc[i, 'Neutral'] = substitute_by_keyboard(dataNew.loc[i, 'Neutral'])[0]
    data = pd.concat([dataOld, dataNew], axis=0, ignore_index=True)
    data = data[['Trump', 'Neutral']]
    data.to_csv(out_path, encoding='utf-8', index=False)
