from tqdm import trange
from IPython.display import HTML, display

import torch.quantization
import torch.nn as nn
import pandas as pd
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor
import time
import warnings

warnings.filterwarnings('ignore')

train_df = pd.read_csv('webNLG2020_train.csv', index_col=[0])

# %%
"""
Trimming off a few data points and so that a batch would not leave any remainder, hence some lines of codes can be avoided (Okay, this might be a hackish way of doing it )
"""

# %%
train_df = train_df.iloc[:35000, :]

# %%
train_df = train_df.sample(frac=1)

# %%
batch_size = 8
num_of_batches = len(train_df) / batch_size
num_of_epochs = 4
train = 1
num_of_batches = int(num_of_batches)

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


def progress(loss, value, max=100):
    return HTML(""" Batch loss :{loss}
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(loss=loss, value=value, max=max))


# %%
num_of_epochs = 1

# %%
"""
## Training the model
"""

# %%
# Sets the module in training mode
if train == 1:
    model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
    # moving the model to device(GPU/CPU)
    model.to(dev)

    """
    ## Initializing the Adafactor optimizer with parameter values suggested for t5
    """

    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    model.train()

    loss_per_10_steps = []
    for epoch in trange(num_of_epochs + 1, desc="'Epoch"):
        print('Running epoch: {}'.format(epoch + 1))

        running_loss = 0

        # out = display(progress(1, num_of_batches + 1), display_id=True)
        for i in trange(num_of_batches, desc="'Batch"):
            inputbatch = []
            labelbatch = []
            new_df = train_df[i * batch_size:i * batch_size + batch_size]
            for indx, row in new_df.iterrows():
                input = 'WebNLG: ' + row['input_text'] + '</s>'
                labels = row['target_text'] + '</s>'
                inputbatch.append(input)
                labelbatch.append(labels)
            if i == 0:
                print(inputbatch, labelbatch)
                exit(1)
            inputbatch = tokenizer.batch_encode_plus(inputbatch, padding=True, max_length=400, return_tensors='pt')[
                "input_ids"]
            labelbatch = tokenizer.batch_encode_plus(labelbatch, padding=True, max_length=400, return_tensors="pt")[
                "input_ids"]
            inputbatch = inputbatch.to(dev)
            labelbatch = labelbatch.to(dev)

            # clear out the gradients of all Variables
            optimizer.zero_grad()

            # Forward propogation
            outputs = model(input_ids=inputbatch, labels=labelbatch)
            loss = outputs.loss
            loss_num = loss.item()
            logits = outputs.logits
            running_loss += loss_num
            if i % 10 == 0:
                loss_per_10_steps.append(loss_num)
            # out.update(progress(loss_num, i, num_of_batches + 1))

            # calculating the gradients
            loss.backward()

            # updating the params
            optimizer.step()

        running_loss = running_loss / int(num_of_batches)
        print('Epoch: {} , Running loss: {}'.format(epoch + 1, running_loss))

    torch.save(model.state_dict(), 'pytoch_model.bin')
    # %%
    """
    ## Plotting the loss over time
    """

    # %%
    import matplotlib.pyplot as plt

    steps = [i * 100 for i in range(len(loss_per_10_steps))]

    plt.plot(steps, loss_per_10_steps)
    plt.title('Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()

    # %%
    """
    ## Testing the model
    """

# %%
model = T5ForConditionalGeneration.from_pretrained('pytoch_model.bin', return_dict=True, config='t5-base-config.json')
model.to(dev)
model.eval()
input_ids = tokenizer.encode("WebNLG: sidharth | hometown | Delhi && sidharth | play |  football </s>",
                             return_tensors="pt")  # Batch size 1
input_ids = input_ids.to(dev)
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

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
    print('Generated in {} seconds'.format(str(elapsed)[:4]))

    return gen_text


"""
# Now, Lets test it out !
"""

# %%
print(generate(' Russia | leader | Putin'))

# %%
print(generate('Sidhath | profession | Doctor  && Sidharth | home_town |  Bombay'))

# %%
print(generate('Nie_Haisheng | birthDate | 1964-10-13  && Nie_Haisheng | occupation | Fighter_pilot '))

# %%
print(generate('Bananaman | creator | Steve_Bright &&  Bananaman | broadcastedBy | BBC'))

# %%
print(generate('Bananaman | lastAired | "1986-04-15" && Bananaman | creator | Steve_Bright'))

# %%
print(generate(
    'Alan_B._Miller_Hall | currentTenants | Mason_School_of_Business && Alan_B._Miller_Hall | location | Williamsburg,_Virginia'))

"""
## Results after quantization
"""

model = model.to("cpu")
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Dropout, nn.LayerNorm}, dtype=torch.qint8
)
model = model.to(dev)

# %%
"""
Lets check the difference in size of the model
"""


# %%
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


print_size_of_model(model)
print_size_of_model(quantized_model)

def quant_generate(text):
    # quantized_model.to(dev)
    quantized_model.eval()
    input_ids = tokenizer.encode("WebNLG:{} </s>".format(text), return_tensors="pt")  # Batch size 1
    # input_ids = input_ids.to(dev)
    s = time.time()
    outputs = quantized_model.generate(input_ids)
    gen_text = tokenizer.decode(outputs[0]).replace('<pad>', '').replace('</s>', '')
    elapsed = time.time() - s
    print('Generated in {} seconds'.format(str(elapsed)[:4]))

    return gen_text

"""
Now lets check the difference in inference time
"""

# %%
print(quant_generate('Facebook | CEO | Mark  && Facebook | number Of Employees | 52000 '))

# %%
print(generate('Facebook | CEO | Mark  && Facebook | number Of Employees | 52000 '))

# %%
