import sys

import pandas as pd

dataset = sys.argv[1]
eval_task = sys.argv[2]
phr_sen = "" if sys.argv[3] == 'sentence' else '_phrase'
run = sys.argv[4]
best_model = True if sys.argv[5] == "True" else False
model_size = "base"

# best_model = True
# best_model = False

# point_gen = True
point_gen = False

print(f"dataset: {dataset}\ntask: {eval_task}\nphr_sen: {phr_sen}\nrun: {run}")
dir_prefix = f"{'PG_' if point_gen else ''}{dataset}{phr_sen}"
conv_pred0 = pd.read_csv(f"results/{eval_task}{run}_{dir_prefix}/{'best_model/' if best_model else ''}converted_predictions0.txt", sep="\t")
conv_pred1 = pd.read_csv(f"results/{eval_task}{run}_{dir_prefix}/{'best_model/' if best_model else ''}converted_predictions1.txt", sep="\t")
conv_pred2 = pd.read_csv(f"results/{eval_task}{run}_{dir_prefix}/{'best_model/' if best_model else ''}converted_predictions2.txt", sep="\t")
# conv_pred0 = pd.read_csv(f"results/{eval_task}{run}/converted_predictions0.txt", sep="\t")
# conv_pred1 = pd.read_csv(f"results/{eval_task}{run}/converted_predictions1.txt", sep="\t")
# conv_pred2 = pd.read_csv(f"results/{eval_task}{run}/converted_predictions2.txt", sep="\t")

new_yn = []
for yn0, yn1, yn2 in zip(conv_pred0["yes_not_pred"].tolist(),
                         conv_pred1["yes_not_pred"].tolist(),
                         conv_pred2["yes_not_pred"].tolist()):
    if yn0 == 1 or yn1 == 1 or yn2 == 1:
        new_yn.append(1)
    else:
        new_yn.append(0)

new_ner = []
for ner0, ner1, ner2 in zip(conv_pred0["predict_ner"].tolist(),
                         conv_pred1["predict_ner"].tolist(),
                         conv_pred2["predict_ner"].tolist()):
    sent_ner = []
    for tag0, tag1, tag2 in zip(ner0.split(), ner1.split(), ner2.split()):
        if tag0 == "[CLS]" or tag1 == "[CLS]" or tag2 == "[CLS]":
            sent_ner.append("[CLS]")
        elif tag0 == "T" or tag1 == "T" or tag2 == "T":
            sent_ner.append("T")
        else:
            sent_ner.append("O")
    new_ner.append(" ".join(sent_ner))

assert len(new_yn) == len(new_ner)
assert len(new_yn) == len(conv_pred0["yes_not"].tolist())

df = pd.DataFrame(conv_pred0["yes_not"], columns=["yes_not"])
df["yes_not_pred"] = new_yn
df["sentence"] = conv_pred0["sentence"]
df["true_ner"] = conv_pred0["true_ner"]
df["predict_ner"] = new_ner

df.to_csv(f"results/{eval_task}{run}_{dir_prefix}/{'best_model/' if best_model else ''}converted_predictions4.txt", sep="\t", index=False, header=True)


