import pandas as pd
import json

with open("sentihood-train.json", "r") as tr, open("sentihood-dev.json", "r") as val, open("sentihood-test.json",
                                                                                             "r") as te:
    train = json.load(tr)
    valid = json.load(val)
    test = json.load(te)


def conv_to_text_gen(data, partition):
    input_text = []
    target_text = []
    for sentence in data:
        opinions = []
        for opinion in sentence["opinions"]:
            # if opinion["aspect"] in ["general", "transit-location", "price", "safety"]:
            # opinions.append(
            #     f"[{opinion['sentiment']}] opinion on [{opinion['aspect']}] for [{opinion['target_entity']}]")
            if task == 'ASD':
                if sent_type == 'sentence':
                    opinions.append(
                        f"[{opinion['sentiment']}] opinion on [{opinion['target_entity']} - {opinion['aspect']}]")
                else:
                    opinions.append(
                        f"{opinion['target_entity']} - {opinion['aspect']} ~ {opinion['sentiment']}")
            else:
                if sent_type == 'sentence':
                    opinions.append(
                        f"opinion on [{opinion['target_entity']} - {opinion['aspect']}]")
                else:
                    opinions.append(
                        f"{opinion['target_entity']} - {opinion['aspect']}")
        opinions = list(set(opinions))
        if len(opinions) > 0:
            input_text.append(sentence["text"].strip())
            if sent_type == 'sentence':
                target_text.append(f"The review expressed {' , '.join(opinions)}".strip())
            else:
                target_text.append(f"{' ~~ '.join(opinions)}".strip())

    df = pd.DataFrame(input_text, columns=["input_text"])
    df["target_text"] = target_text
    df["prefix"] = [f"{task}"] * len(target_text)
    if sent_type == 'sentence':
        df.to_csv(f"{partition}_{task}.csv", header=True, index=False)
    else:
        df.to_csv(f"{partition}_{task}_phrase.csv", header=True, index=False)


for task in ['ASD', 'AD']:
    for sent_type in ['sentence', 'phrase']:
        conv_to_text_gen(train, "train")
        conv_to_text_gen(valid, "valid")
        conv_to_text_gen(test, "test")

