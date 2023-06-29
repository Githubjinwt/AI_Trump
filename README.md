# AI_Trump
AI Trump: A tool Transferring your sentences to Trump-style  

## download model and checkpoint
save models and checkpoints under the folder 'model'
|name|link(百度网盘)|
|--|--|
| model1 | [提取码: 4coz](https://pan.baidu.com/s/1VyMav4Wt-ZENUzMr1U_tiQ) |
| checkpoint1 | [提取码: 0s8e](https://pan.baidu.com/s/1hCVp3UprTT9eFbVwMEdKRQ) |
| model2 | [提取码: 7jp3](https://pan.baidu.com/s/1O8pGXvDngu9E3krPYojQpQ) |
| checkpoint2 | [提取码: ioxy](https://pan.baidu.com/s/1EFV95ve2xUtaa-rBTsiK3w) |

## Prerequisites
Create several folders:
```bash
mkdir model
mkdir prediction
```
Install required modules and tools:
```bash
pip install datasets evaluate gradio semantic_text_similarity flair
```

## Launching Demo Locally
Try out our demo [demo.py](demo.py) on your local machine by running
```bash
bash bash/demo.sh
```
and modify parameters as you need.

## Training
Set training parameters in [train.sh](bash/train.sh) and train your own model by running
```bash
bash bash/train.sh
```

## Evaluating
Evaluate your trained model by running
```bash
bash bash/eval.sh
```

## Acknowledgement
The research work comes from the final project of Deep Learning class by Prof. Jing Zhang (RUC). We would like to thank the professor and her assistants for their great effort, and also thank our reviewers (classmates) for their valuable comments. 

