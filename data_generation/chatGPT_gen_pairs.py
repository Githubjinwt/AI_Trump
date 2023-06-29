# generate data pairs using OpenAI ChatGPT

import openai
import pandas as pd
import tqdm
from time import sleep

chat_gpt_key = ''
openai.api_key = chat_gpt_key

# GLM_API_key = ""
# GLM_Public_key = ""

def completion(prompt):
    # davinci model, more expensive
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=1024,
        n=1,
        stop=None
    )
    message = response.choices[0].text
    return message

def ChatCompletion(msg):
    # chat mode, using turbo, cheaper but behaving worse
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": msg}
        ]
    )
    message = response['choices'][0]['message']['content']
    return message


promptTrump2Neutral = '''
The following sentence was spoken by former US President Trump,
and now you need to transfer it to a neutral tone,
and just reply the answer with nothing else: '{}'.
'''

# this_msg = "The Russia Hoax is the biggest political scandal in American history. Treason!!! Lets see how it ends????_link_…"
# davinci: The Russia Hoax has been described as one of the most significant political scandals in American history. It remains to be seen how it will conclude.
# chat: A significant political scandal in American history is the Russia Hoax. The outcome remains to be seen.

in_path = ""
out_path = ""

data = pd.read_csv(in_path, encoding='utf-8')

for i in tqdm.tqdm(range(0, len(data))):
    data.loc[i, 'Neutral'] = ChatCompletion(msg=promptTrump2Neutral.format(data.loc[i, 'Trump']))
    # save every step, in case disconnection
    data[['Trump', 'Neutral']].to_csv(out_path, encoding='utf-8', index=False)
    # one minute three times limitation
    sleep(20)

# print(completion(prompt=promptTrump2Neutral.format(this_msg)))
# print(ChatCompletion(msg=promptTrump2Neutral.format(this_msg)))
