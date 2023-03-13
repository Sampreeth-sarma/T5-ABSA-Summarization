import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import warnings

warnings.filterwarnings('ignore')

dataset = 'semeval-2016'
task = 'TASD'

test_df = pd.read_csv(f'data/{dataset}/test_{task}.csv')
# %%
batch_size = 8
num_of_epochs = 50

if torch.cuda.is_available():
    dev = torch.device("cuda:6")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

# %%
"""
## Loading the pretrained model and tokenizer
"""

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(f'results/{dataset}_{task}_{num_of_epochs}.bin', return_dict=True,
                                                   config='t5-base-config.json')
model.to(dev)
model.eval()

"""
## The Inference function
"""


def generate(text):
    model.eval()
    input_ids = tokenizer.encode("WebNLG:{} </s>".format(text), return_tensors="pt")  # Batch size 1
    input_ids = input_ids.to(dev)
    s = time.time()
    outputs = model.generate(input_ids).to(dev)
    gen_text = tokenizer.decode(outputs[0]).replace('<pad>', '').replace('</s>', '')
    elapsed = time.time() - s
    # print('Generated in {} seconds'.format(str(elapsed)[:4]))

    return gen_text


input = 'Semeval: ' + 'Nice ambience , but highly overrated place .' + '</s'
print(f"input:{input}\nmodel output: {generate(input)}")

input = 'Semeval: ' + 'Nice ambience' + '</s'
print(f"input:{input}\nmodel output: {generate(input)}")

input = 'Semeval: ' + 'but highly overrated place .' + '</s'
print(f"input:{input}\nmodel output: {generate(input)}")

input = 'Semeval: ' + 'The coffe is very good , too .' + '</s'
print(f"input:{input}\nmodel output: {generate(input)}")

input = 'Semeval: ' + 'The coffee is very good , too .' + '</s'
print(f"input:{input}\nmodel output: {generate(input)}")

input = 'Semeval: ' + "The restaurant offers an extensive wine list and an ambiance you won ' t forget !" + '</s'
print(f"input:{input}\nmodel output: {generate(input)}")

input = 'Semeval: ' + "The restaurant offers an extensive wine list" + '</s'
print(f"input:{input}\nmodel output: {generate(input)}")

input = 'Semeval: ' + 'The coffee is very good , too .' + '</s'
print(f"input:{input}\nmodel output: {generate(input)}")

