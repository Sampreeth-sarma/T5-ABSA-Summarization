import pandas as pd
import sys
import spacy

# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# task = "TASD"
# task = sys.argv[1]


def get_opinion_string(task, polarity, aspect_category, target_words):
    if task == 'TASD':
        return f"[{polarity}] opinion on [{aspect_category}] for [{target_words.strip()}]"
    elif task == 'AD':
        return f"opinion on [{aspect_category}]"
    elif task == 'TD':
        return f"opinion for [{target_words.strip()}]"
    elif task == 'ASD':
        return f"[{polarity}] opinion on [{aspect_category}]"
    elif task == 'TSD':
        return f"[{polarity}] opinion for [{target_words.strip()}]"
    elif task == 'TAD':
        return f"opinion on [{aspect_category}] for [{target_words.strip()}]"
    else:
        return ""


for task in ['TASD']:
    for partition in ["train", "test"]:
        input_text_ids, input_text, target_text = [], [], []
        with open(f'{partition}_TAS.tsv', "r") as fd:
            fd.readline()
            lines = fd.readlines()

        seg_a, seg_b, is_target = [], [], []

        for i in range(0, len(lines), 36):
            sentence = lines[i: (i + 36)]

            # check whether this chunk belongs to only one sentence
            assert len(set([each.split("\t")[3] for each in sentence])) == 1

            # append the sentence and the sentence id into the respective lists
            input_text_ids.append(sentence[0].split("\t")[0].strip())
            input_text.append(sentence[0].split("\t")[3].strip())

            # added for the noun_target
            spacy_doc = nlp(sentence[0].split("\t")[3].strip())
            noun_list = set()
            for each_doc in spacy_doc:
                if each_doc.pos_ in ["NOUN", "PROPN"]:
                    noun_list.add(str(each_doc.text))
            noun_list = list(noun_list)
            # #######

            opinion_list = []
            for each_tas in sentence:
                each_tas_parts = each_tas.split("\t")
                if each_tas_parts[1] == "1":
                    aspect_category = ' '.join(each_tas_parts[2].split()[:2]).strip()
                    polarity = each_tas_parts[2].split()[-1].strip()
                    target_ner_tags = each_tas_parts[4].split()
                    sent_words = each_tas_parts[3].split()
                    target_one_indices = [idx for idx, each_tag in enumerate(target_ner_tags) if each_tag != 'O']
                    tgt_idx = 0
                    target_words = ""
                    target_words_list = []
                    while tgt_idx < len(target_one_indices):
                        if tgt_idx == 0:
                            target_words = sent_words[target_one_indices[tgt_idx]]
                        elif target_one_indices[tgt_idx] == target_one_indices[tgt_idx - 1] + 1:
                            target_words = target_words + " " + sent_words[target_one_indices[tgt_idx]]
                            # target_words = ' '.join([sent_words[each_target_one_idx] for each_target_one_idx in target_one_indices]).strip()
                        else:
                            if target_words == '':
                                target_words = "NULL"
                            # opinion_list.append(f"[{polarity}] opinion on [{aspect_category}] for [{target_words}]")
                            opinion_list.append(get_opinion_string(task, polarity, aspect_category, target_words))
                            target_words = sent_words[target_one_indices[tgt_idx]]
                        tgt_idx += 1
                    if target_words == '':
                        target_words = "NULL"
                    # opinion_list.append(f"[{polarity}] opinion on [{aspect_category}] for [{target_words.strip()}]")
                    opinion_list.append(get_opinion_string(task, polarity, aspect_category, target_words))

            # to build the auxiliary sentence that is used as the output of the text generation model
            auxiliary_text = "The review expressed " + ", ".join(opinion_list)
            target_text.append(auxiliary_text)

            # added for noun_target
            target_list = set()
            for op in opinion_list:
                target_list.add(op.split(" for [")[1][:-1])

            seg_a.extend([input_text[-1]] * len(noun_list))
            seg_b.extend(noun_list)
            for each_noun in noun_list:
                flag = 0
                if each_noun in target_list:
                    flag = 1
                else:
                    for each_target in target_list:
                        if each_noun in each_target:
                            flag = 1
                            break
                if flag == 1:
                    is_target.append(1)
                else:
                    is_target.append(0)
            # #####

        df = pd.DataFrame(seg_a, columns=["text_a"])
        df["text_b"] = seg_b
        df["labels"] = is_target
        df.to_csv(f"{task}_{partition}_noun_target.csv", header=True, index=False)
