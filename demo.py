# download the model and checkpoint, then try AI_Trump local!!

import gradio as gr

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model1 = AutoModelForSeq2SeqLM.from_pretrained("model1 path")
tokenizer1 = AutoTokenizer.from_pretrained("checkpoint1 path")

model2 = AutoModelForSeq2SeqLM.from_pretrained("model2 path")
tokenizer2 = AutoTokenizer.from_pretrained("checkpoint2 path")

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
    inputs=gr.Textbox(label="Input Neutral words here"),
    outputs=gr.TextArea(label="Transfered words"),
    examples=[
        ["abc"],
        ["Make America great again."],
    ],
    title="Text Style Transfer to Trump",
    description="Here's a sample transfering text style to Trump-Style . Enjoy!",
    article = "Check out the examples"
)
demo.launch(share=True)
