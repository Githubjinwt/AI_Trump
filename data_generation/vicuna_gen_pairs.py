# generate data pairs using vicuna locally

from transformers import GenerationConfig
import torch
from fastchat.model import load_model, get_conversation_template
import pandas as pd
import tqdm

max_new_tokens = 512
temperature = 1.0
top_p = 1.0
model_path = 'your_vicuna_weight_path'

model, tokenizer = load_model(
	model_path=model_path,
	device="cuda",
	num_gpus=1,
)

def check(msg):
    msg = '''
    Check whether the following sentence is said by former US President Donald Trump: "{}",
    and just answer "Yes" or "No" with nothing else.
    '''.format(msg)
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        inputs=torch.as_tensor(input_ids).cuda(),
        generation_config=GenerationConfig(
			do_sample=True,
			temperature=temperature,
			max_new_tokens=max_new_tokens
        )
    )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs

def transfer(msg):
    msg = '''
    The following sentence was spoken by former US President Trump,
    and now you need to transfer it to a normal style: "{}".
    Just give me the transfered sentence.
    '''.format(msg)

    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        inputs=torch.as_tensor(input_ids).cuda(),
        generation_config=GenerationConfig(
			do_sample=True,
			temperature=temperature,
			max_new_tokens=max_new_tokens
        )
    )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs

# check whether it is said by Trump
# acc = 0.6
# def check_mul_times(msg, times=10):
#     y_times = 0
#     for i in range(times):
#         res = check(msg)
#         if "Yes" in res or "yes" in msg:
#             y_times += 1
#     return y_times/times >= acc

if __name__ == "__main__":
    dataOld = pd.read_csv('speeches.csv', encoding='utf-8')
    dataNew = pd.DataFrame()
    # for every sentence, we repeat some times
    repeat = 2
    pos = 0
    for i in tqdm.tqdm(range(len(dataOld))):
        for j in range(repeat):
            cnt = dataOld.loc[i, 'Trump']
            dataNew.loc[pos+j, 'Trump'] = cnt
            dataNew.loc[pos+j, 'Neutral'] = transfer(cnt)
        pos += repeat
        # if check_mul_times(dataOld.loc[i, 'content']):
    dataNew.to_csv('vicuna.csv', encoding='utf-8')
