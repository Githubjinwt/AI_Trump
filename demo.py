# download the model and checkpoint, and try AI_Trump local!!

import gradio as gr
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for demo.")
    parser.add_argument('--model1_path', type = str, default = "model/model1",
                        help = 'model1 saving path.')
    parser.add_argument('--model2_path', type = str, default = "model/model2",
                        help = 'model2 saving path.')
    parser.add_argument('--checkpoint1_path', type = str, default = "model/checkpoint1",
                        help = 'checkpoint1 saving path.')
    parser.add_argument('--checkpoint2_path', type = str, default = "model/checkpoint2",
                        help = 'checkpoint2 saving path.') 
    opt = parser.parse_args()
    return opt

opt = parse_option()

model1 = AutoModelForSeq2SeqLM.from_pretrained(opt.model1_path)
tokenizer1 = AutoTokenizer.from_pretrained(opt.checkpoint1_path)

model2 = AutoModelForSeq2SeqLM.from_pretrained(opt.model2_path)
tokenizer2 = AutoTokenizer.from_pretrained(opt.checkpoint2_path)

# model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
# tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')

def transfer(Input):
    input_ids = tokenizer1.batch_encode_plus([Input], max_length=1024, return_tensors='pt', truncation=True)['input_ids']
    output_ids = model1.generate(input_ids, num_beams=1, length_penalty=2, max_length=100, min_length=5, no_repeat_ngram_size=3)
    output_txt = tokenizer1.decode(output_ids.squeeze(), skip_special_tokens=True)

    input_ids = tokenizer2.batch_encode_plus([output_txt], max_length=1024, return_tensors='pt', truncation=True)['input_ids']
    output_ids = model2.generate(input_ids, num_beams=1, length_penalty=2, max_length=100, min_length=5, no_repeat_ngram_size=3)
    output_txt = tokenizer2.decode(output_ids.squeeze(), skip_special_tokens=True)
    return output_txt

demo = gr.Interface(
    fn=transfer,
    # 设置输入
    inputs=gr.Textbox(label="Input Neutral sentences here"),
    # 设置输出
    outputs=gr.TextArea(label="Transfered sentences"),
    # 设置输入参数示例
    examples=[
        ["We can make America stronger."],
        ["China."],
        ["I'm Donald Trump."],
        ["I like machine learning."],
        ["Democrats are trying to impeach President Trump because they are jealous of his success."],
        ["You're out of luck."],
        ["Campaign finance reform is an important issue that affects the integrity of our democracy."],
        ["We need law enforcement to keep order."],
        ["The ongoing debate over the controversial issue of gun control is a complex and delicate matter that requires careful consideration and negotiation in order to find a viable solution."],
        ["Fuck you."]
    ],
    # 设置网页标题
    title="AI Trump(赛博川普): Text Style Transfer to Trump",
    # 左上角的描述文字
    description="Here's a web transfering your sentence style to Trump-Style. Enjoy!",
    # 左下角的文字
    article = "Check out the examples"
)
demo.launch(share=True)
