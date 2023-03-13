import pandas as pd
import sys

dataset = sys.argv[1]
eval_task = sys.argv[2]
phr_sen = "" if sys.argv[3] == 'sentence' else '_phrase'
run = sys.argv[4]

df = pd.read_csv(f'data/{dataset}/test_{eval_task}{phr_sen}.csv')
test_sents = df["input_text"].values.tolist()

user_inp = input("pretrained model dataset: Enter '0' for MinWiki, '1' for DeSSE, or any number for both\n")
model_dataset = "MinWiki" if user_inp == "0" else "DeSSE" if user_inp == 1 else ["MinWiki", "DeSSE"]

model = T5ForConditionalGeneration.from_pretrained(f"../ABSA_Datasets/Split_Sentences/model_files/{model_dataset}")
model = model.to(device)
tokenizer = T5Tokenizer.from_pretrained(f"../ABSA_Datasets/Split_Sentences/model_files/{model_dataset}")

test_df = pd.DataFrame({'Text': test_sents, 'Split_Sentences': [''] * len(test_sents)})

model_params["TRAIN_SIZE"] = 0.001
_, test_loader = get_data_loaders(test_df, "Text", "Split_Sentences", tokenizer, model_params)
predictions, actuals, input_texts = test(0, tokenizer, model, device, test_loader)

out = "\n".join([f"{inp}\n{pr}\n" for inp, pr in zip(input_texts, predictions)])
with open(f"Split_Sentences/outputs/T5_pretrained_{model_dataset}_test_MAMS_{test_dataset}_output.txt", "w+") as f:
    f.write(out)