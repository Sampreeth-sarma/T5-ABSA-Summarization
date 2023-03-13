import os
import pickle
import re
import sys
import warnings
import xml.dom.minidom as DOM
import xml.etree.ElementTree as ET
from datetime import datetime
from statistics import mean

import pandas as pd
import torch.cuda
from scipy.stats import pearsonr, spearmanr
from simpletransformers.t5 import T5Model
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1
import spacy

nlp = spacy.load("en_core_web_sm")

warnings.filterwarnings('ignore')


def f1(truths, preds):
    return mean([compute_f1(truth, pred) for truth, pred in zip(truths, preds)])


def exact(truths, preds):
    return mean([compute_exact(truth, pred) for truth, pred in zip(truths, preds)])


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?.$*+;/:@&#%\"=\-'`–’é]", " ", string)
    # string = " ".join(re.split("[^a-zA-Z]", string.lower())).strip()
    string = re.sub(r"\'s", " \' s", string)
    string = re.sub(r"\'ve", " \' ve", string)
    string = re.sub(r"\'t", " \' t", string)
    string = re.sub(r"\'re", " \' re", string)
    string = re.sub(r"\'d", " \' d", string)
    string = re.sub(r"\'ll", " \' ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\+", " + ", string)
    string = re.sub(r"\$", " $ ", string)
    string = re.sub(r"\*", " * ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"-", " - ", string)
    string = re.sub(r"\;", " ; ", string)
    string = re.sub(r"\/", " / ", string)
    string = re.sub(r"\:", " : ", string)
    string = re.sub(r"\@", " @ ", string)
    string = re.sub(r"\#", " # ", string)
    string = re.sub(r"\%", " % ", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"\&", " & ", string)
    string = re.sub(r"=", " = ", string)
    string = re.sub(r"–", " – ", string)
    string = re.sub(r"’", " ’ ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def longestCommonPrefix(strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    if len(strs) == 0:
        return ""
    current = strs[0]
    for i in range(1, len(strs)):
        temp = ""
        if len(current) == 0:
            break
        for j in range(len(strs[i])):
            if j < len(current) and current[j] == strs[i][j]:
                temp += current[j]
            else:
                break
        current = temp
    return current


def compute_F1_for_TSD(true_aspect_idx, pred_aspect_idx, true_pol, pred_pol, implicit):
    true_tgt_pol, pred_tgt_pol, common_tgt_pol = set(), set(), set()
    for t_pol, t_tgt in zip(true_pol, true_aspect_idx):
        if len(t_tgt) > 0:
            # for not NULL cases (explicit targets)
            true_tgt_pol.add(f"{t_tgt[0]} - {t_tgt[-1]} ~ {t_pol}")
        if implicit and len(t_tgt) == 0:
            # for NULL cases (implicit targets)
            true_tgt_pol.add(f"0 - 0 ~ {t_pol}")

    for p_pol, p_tgt in zip(pred_pol, pred_aspect_idx):
        if len(p_tgt) > 0:
            pred_tgt_pol.add(f"{p_tgt[0]} - {p_tgt[-1]} ~ {p_pol}")
        if implicit and len(p_tgt) == 0:
            pred_tgt_pol.add(f"0 - 0 ~ {p_pol}")

    common_tgt_pol = true_tgt_pol & pred_tgt_pol

    return true_tgt_pol, pred_tgt_pol, common_tgt_pol


def convert_pred_to_TAS_format(truth, preds, gold_xml_file=None, input_text_col="input_text"):
    new_preds = []
    trimmed_preds = []
    num_trimmed_sentences = 0
    for pred in preds:
        new_pred = []
        trim_flag = True
        for p in pred:
            if phr_sen == '':
                if eval_task == 'ASD':
                    match = re.match(
                        r"(The review expressed (\[(positive|negative|neutral)\] opinion on \[(.+?)\](, )*)+)", p)
                else:
                    match = re.match(
                        r"(The review expressed (opinion on \[(.+?)\](, )*)+)", p)
            else:
                if eval_task == 'ASD':
                    match = re.match(r"((.*(\s\s)(positive|negative|neutral)(\s)*)+)", p)
                else:
                    match = re.match(r"(([a-z]+(\s)[a-z]+(\s\s)*)+)", p)
            if match:
                out = match.groups()[0].strip().strip(",")
                new_pred.append(out)
            else:
                new_pred.append("")

            if p != new_pred[-1]:
                if new_pred[-1] == "":
                    trimmed_preds.append("--------------")
                else:
                    trimmed_preds.append(p + "\nchanged pred: \n" + new_pred[-1])
                if trim_flag:
                    trim_flag = False
                    num_trimmed_sentences += 1
        new_preds.append(new_pred)

    with open('trimmed_preds1.txt', "w+") as f:
        f.write(f"Number of trimmed sentences={num_trimmed_sentences}\n\n")
        f.write("\n\n".join(trimmed_preds))

    preds = new_preds

    if not os.path.exists("predictions"):
        os.mkdir("predictions")

    # Saving the predictions if needed
    with open(f"predictions/{eval_task}_{dir_prefix}_predictions_{datetime.now()}.txt", "w") as f:
        for i, text in enumerate(df[input_text_col].tolist()):
            f.write(str(text) + "\n\n")

            f.write("Truth:\n")
            f.write(truth[i] + "\n\n")

            f.write("Prediction:\n")
            for pred in preds[i]:
                f.write(str(pred) + "\n")
            f.write("________________________________________________________________________________\n")

    # exit(1)

    def getsubidx(x, y):
        l1, l2 = len(x), len(y)
        for i in range(l1):
            if x[i:i + l2] == y:
                return i
        return -1

    # get the gold annotations for the aspect-sentiment, yes_no, ner_tags from the TAS-BERT test file
    gold_df = pd.read_csv(f'data/{dataset}/test_TAS.tsv', sep="\t")
    gold_id = gold_df["sentence_id"].tolist()

    for pred_offset in range(3):
        # get the input text ids, and input text from the text_gen test set for this task
        input_text = df[input_text_col].tolist()
        dup_count = 0
        longest_prefix_count = 0

        # clear the gold opinions and get the empty framework
        sen_tree_map = {}
        xml_tree = ET.parse(gold_xml_file)
        root = xml_tree.getroot()

        for node in root.iter('Review'):
            for sen in node.iter('sentence'):
                for elem in sen.iter():
                    if elem.tag == 'sentence':
                        sen_key = elem.attrib['id']
                        sen_tree_map[sen_key] = sen
                    if elem.tag == 'Opinions':
                        if elem is not None:
                            elem.clear()

        Common_Num = 0
        True_Num = 0
        Pred_Num = 0
        true_asp_pol_list, pred_asp_pol_list = [], []
        for idx, inp_text in enumerate(input_text):
            wrong_flag = False
            num_combinations = 36 if dataset == 'semeval-2016' else 39
            sentence_id = list(set(gold_id[idx * num_combinations: (idx + 1) * num_combinations]))

            assert len(sentence_id) == 1, "************ 2 different sentence ids ***************"
            sentence_id = sentence_id[0]

            current_sen = sen_tree_map[sentence_id]
            current_opinions = current_sen.find('Opinions')
            if current_opinions == None:
                current_opinions = ET.Element('Opinions')
                current_sen.append(current_opinions)

            # extract true and predicted aspect categories adn the polarities
            if phr_sen == '':
                true_aspect = re.findall(r"opinion on \[(.+?)\]", truth[idx])
                pred_aspect = re.findall(r"opinion on \[(.+?)\]", preds[idx][pred_offset])
            else:
                true_aspect = [each_op.split(" ~ ")[0] for each_op in truth[idx].split(" ~~ ")]
                pred_aspect = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  ")) if
                               op_idx % 2 == 0 and preds[idx][pred_offset] != '']

            true_aspect = ["#".join(each_asp.upper().split()) for each_asp in true_aspect]
            pred_aspect = ["#".join(each_asp.upper().split()) for each_asp in pred_aspect]

            if eval_task == 'ASD':
                if phr_sen == '':
                    true_pol = re.findall(r" \[([A-Za-z]+)\] opinion on", truth[idx])
                    pred_pol = re.findall(r" \[([A-Za-z]+)\] opinion on", preds[idx][pred_offset])
                else:
                    true_pol = [each_op.split(" ~ ")[1] for each_op in truth[idx].split(" ~~ ")]
                    pred_pol = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  ")) if
                                op_idx % 2 == 1 and preds[idx][pred_offset] != '']

                # If any aspect polarity is dropped by any chance, then, we have to exclude that respective
                # target also
                if len(pred_pol) != len(pred_aspect):
                    pred_aspect = pred_aspect[:len(pred_pol)]

                assert len(true_pol) == len(true_aspect)
                assert len(pred_pol) == len(pred_aspect)

                true_asp_pol, pred_asp_pol, common_asp_pol = set(), set(), set()
                for t_pol, t_asp in zip(true_pol, true_aspect):
                    true_asp_pol.add(f"{t_asp} ~ {t_pol}")
                for p_pol, p_asp in zip(pred_pol, pred_aspect):
                    pred_asp_pol.add(f"{p_asp} ~ {p_pol}")
                pred_asp_pol_list.append(inp_text + "\nTruth: \n" + str(sorted(list(true_asp_pol))) + "\nPred: \n" + str(sorted(list(pred_asp_pol))))

                commmon_asp_pol = true_asp_pol & pred_asp_pol
                True_Num += len(true_asp_pol)
                Pred_Num += len(pred_asp_pol)
                Common_Num += len(commmon_asp_pol)

            # to generate the XML file for AD evaluation

            for each_asp in set(pred_aspect):
                op = ET.Element('Opinion')
                op.set('target', '')
                op.set('category', each_asp)
                op.set('polarity', "")
                op.set('from', '0')
                op.set('to', '0')
                current_opinions.append(op)

        if eval_task == 'ASD':
            P = Common_Num / float(Pred_Num) if Pred_Num != 0 else 0
            R = Common_Num / float(True_Num)
            F = (2 * P * R) / float(P + R) if P != 0 else 0

            print('ASD task')
            print("\tP: ", P, "   R: ", R, "  F1: ", F)
            print('----------------------------------------------------\n\n')

        xml_string = ET.tostring(root)
        xml_write = DOM.parseString(xml_string)
        with open(f'evaluation_for_AD_TD_TAD/{eval_task}{run}_{dir_prefix}_sentence{pred_offset}.xml', 'w') as handle:
            xml_write.writexml(handle, indent=' ', encoding='utf-8')
        with open(f"outputs/{eval_task}{run}_{dir_prefix}_sentence{pred_offset}_pred{'_coref' if op_mode == 2 else '_splits' if op_mode == 1 else ''}_ASD.txt", "w+") as pr:
            pr.write("\n\n".join(pred_asp_pol_list))

        print(
            f"\n\n\n*******\nGenarated target XML: {eval_task}{run}_{dir_prefix}_sentence{pred_offset}.xml'\n*********\n\n")


model_args = {
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "eval_batch_size": 8,
    "use_multiprocessing": False,
    "use_multiprocessing_for_evaluation": False,
    "use_multiprocessed_decoding": False,
    "num_beams": None,
    "do_sample": True,
    "max_length": 512,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,
}

# Load the evaluation data

dataset = sys.argv[1]
eval_task = sys.argv[2]
phr_sen = "" if sys.argv[3] == 'sentence' else '_phrase'
run = sys.argv[4]
op_mode = int(sys.argv[5]) if len(sys.argv) == 6 else 0

model_size = "base"
dir_prefix = f"{dataset}{phr_sen}"
print(f"dataset: {dataset}\ntask: {eval_task}\nphr_sen: {phr_sen}\nrun: {run}")

if op_mode == 2:
    input_text_col = "resolved_texts"
    filename = f'data/{dataset}/test_{eval_task}{phr_sen}_splits_Org_Coref_resolved.csv'
elif op_mode == 1:
    input_text_col = "input_text"
    filename = f'data/{dataset}/test_{eval_task}{phr_sen}_split_sents_Org.csv'
elif op_mode == 0:
    input_text_col = "input_text"
    filename = f'data/{dataset}/test_{eval_task}{phr_sen}.csv'
else:
    print("ERROR: invalid input for op_mode")
    exit(1)

print(f"\n___________________ \n{filename}\n{input_text_col}\n____________________\n")
df = pd.read_csv(filename)

tasks = df["prefix"].tolist()
analysis = False
# analysis = True

if not analysis:
    # Load the trained model
    # model = T5Model("t5", "outputs", args=model_args)
    model = T5Model("t5", f"results/{eval_task}{run}_{dir_prefix}/", args=model_args,
                    use_cuda=False if not torch.cuda.is_available() else True)

    # Prepare the data for testing
    if op_mode == 2:
        inp_strs = df["resolved_texts"].values.tolist()
        truth = df["target_text"].tolist()
        idxs, split_sents, org_sents = [], [], []
        count = 0
        for each in inp_strs:
            #     set_trace()
            doc = nlp(each)
            sents = list(doc.sents)
            idxs.extend([count] * len(sents))
            split_sents.extend([str(sent) for sent in sents])
            org_sents.extend([each] * len(sents))
            count += 1

        final_df = pd.DataFrame({"sent_num": idxs, "org_sent": org_sents, "split_sents": split_sents})

        to_predict = ["ASD: " + str(input_text) for input_text in final_df["split_sents"].tolist()]
        print(to_predict[:5])
    else:
        to_predict = [
            prefix + ": " + str(input_text) for prefix, input_text in zip(df["prefix"].tolist(), df[input_text_col].tolist())
        ]
        truth = df["target_text"].tolist()

    # Get the model predictions
    preds = model.predict(to_predict)

    if op_mode == 2:
        preds_reverted = []
        preds_sents = []
        for i, act_sent_num in enumerate(final_df["sent_num"].values.tolist()):
            if i == 0:
                prev_sent_num = 0
                preds_sents.append(preds[i][0])
                preds_sents.append(preds[i][1])
                preds_sents.append(preds[i][2])

            elif prev_sent_num != act_sent_num:
                prev_sent_num += 1
                preds_reverted.append(preds_sents)
                preds_sents = []
                preds_sents.append(preds[i][0])
                preds_sents.append(preds[i][1])
                preds_sents.append(preds[i][2])
            else:
                for sent_pred_num, pred in enumerate(preds[i]):
                    preds_sents[sent_pred_num] += ", " + pred.strip("The review expressed").strip()
        preds_reverted.append(preds_sents)

        assert len(preds_reverted) == len(truth), f"{len(preds_reverted)} : {len(truth)}"
        df["preds_reverted"] = preds_reverted
        df.to_csv("coref_SE-16_preds.csv", header=True, index=False)
        # exit(1)
        preds = preds_reverted

    print("\n".join(preds[0]))

    with open(f'{eval_task}{run}_{dir_prefix}_truth.pkl', "wb") as f:
        pickle.dump(truth, f)
    with open(f'{eval_task}{run}_{dir_prefix}_preds.pkl', "wb") as f:
        pickle.dump(preds, f)

else:
    with open(f'{eval_task}{run}_{dir_prefix}_truth.pkl', "rb") as f:
        truth = pickle.load(f)
    with open(f'{eval_task}{run}_{dir_prefix}_preds.pkl', "rb") as f:
        preds = pickle.load(f)

# Saving the predictions if needed
convert_pred_to_TAS_format(truth, preds,
                           f"evaluation_for_AD_TD_TAD/ABSA{15 if '15' in dataset else 16}_Restaurants_Test.xml",
                           input_text_col=input_text_col
                           )

preds = [pred[0] for pred in preds]
df["predicted"] = preds

output_dict = {each_task: {"truth": [], "preds": [], } for each_task in tasks}
print(output_dict)

results_dict = {}

for task, truth_value, pred in zip(tasks, truth, preds):
    output_dict[task]["truth"].append(truth_value)
    output_dict[task]["preds"].append(pred)
# print(output_dict)

print("-----------------------------------")
print("Results: ")
for task, outputs in output_dict.items():
    if task == f"Semeval {eval_task}" or task == eval_task:
        try:
            task_truth = output_dict[task]["truth"]
            task_preds = output_dict[task]["preds"]
            # print("computing metrics")
            results_dict[task] = {
                "F1 Score": f1(task_truth, task_preds) if (
                        eval_task == "Semeval AD" or eval_task == "Semeval ASD") else "Not Applicable",
                "Exact matches": exact(task_truth, task_preds),
            }
            print(f"Scores for {task}:")
            print(
                f"F1 score: {f1(task_truth, task_preds) if (eval_task == 'Semeval AD' or eval_task == 'Semeval ASD') else 'Not Applicable'}")
            print(f"Exact matches: {exact(task_truth, task_preds)}")
            print()
        except:
            pass

# with open(f"results/result_{datetime.now()}.json", "w") as f:
#     json.dump(results_dict, f)
