# python evaluation_for_TSD_ASD_TASD.py --output_dir TASD_base_EOS --tag_schema TO --num_epochs 0
# python evaluation_for_TSD_ASD_TASD.py --output_dir TASD_base_EOS --tag_schema TO --num_epochs 1
# python evaluation_for_TSD_ASD_TASD.py --output_dir TASD_base_EOS --tag_schema TO --num_epochs 2
# python change_pre_to_xml.py --gold_path ../data/semeval-2016/test_TAS.tsv --pre_path ../TASD_base_/converted_predictions0.txt --gold_xml_file ABSA16_Restaurants_Test.xml --pre_xml_file pred_file_2016_T5_base_0.xml --tag_schema TO
# python change_pre_to_xml.py --gold_path ../data/semeval-2016/test_TAS.tsv --pre_path ../TASD_base_/converted_predictions1.txt --gold_xml_file ABSA16_Restaurants_Test.xml --pre_xml_file pred_file_2016_T5_base_1.xml --tag_schema TO
# python change_pre_to_xml.py --gold_path ../data/semeval-2016/test_TAS.tsv --pre_path ../TASD_base_/converted_predictions2.txt --gold_xml_file ABSA16_Restaurants_Test.xml --pre_xml_file pred_file_2016_T5_base_2.xml --tag_schema TO
# java -cp ./A.jar absa15.Do Eval ./pred_file_2016_T5_base_0.xml ./ABSA16_Restaurants_Test.xml 1 0
# java -cp ./A.jar absa15.Do Eval ./pred_file_2016_T5_base_0.xml ./ABSA16_Restaurants_Test.xml 2 0
# java -cp ./A.jar absa15.Do Eval ./pred_file_2016_T5_base_0.xml ./ABSA16_Restaurants_Test.xml 3 0
# java -cp ./A.jar absa15.Do Eval ./pred_file_2016_T5_base_1.xml ./ABSA16_Restaurants_Test.xml 1 0
# java -cp ./A.jar absa15.Do Eval ./pred_file_2016_T5_base_1.xml ./ABSA16_Restaurants_Test.xml 2 0
# java -cp ./A.jar absa15.Do Eval ./pred_file_2016_T5_base_1.xml ./ABSA16_Restaurants_Test.xml 3 0
# java -cp ./A.jar absa15.Do Eval ./pred_file_2016_T5_base_2.xml ./ABSA16_Restaurants_Test.xml 1 0
# java -cp ./A.jar absa15.Do Eval ./pred_file_2016_T5_base_2.xml ./ABSA16_Restaurants_Test.xml 2 0
# java -cp ./A.jar absa15.Do Eval ./pred_file_2016_T5_base_2.xml ./ABSA16_Restaurants_Test.xml 3 0

import json
import os
import pickle
import re
import sys
import warnings
from datetime import datetime
from statistics import mean

import pandas as pd
import torch.cuda
from scipy.stats import pearsonr, spearmanr
from simpletransformers.t5 import T5Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

warnings.filterwarnings('ignore')


def f1(truths, preds):
    return mean([compute_f1(truth, pred) for truth, pred in zip(truths, preds)])


def exact(truths, preds):
    return mean([compute_exact(truth, pred) for truth, pred in zip(truths, preds)])


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]


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


def sentihood_strict_acc_t5(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of Sentihood.
    """
    total_cases = int(len(y_true) / 8)
    true_cases = 0
    for i in range(total_cases):
        if y_true[i * 8] != y_pred[i * 8]: continue
        if y_true[i * 8 + 1] != y_pred[i * 8 + 1]: continue
        if y_true[i * 8 + 2] != y_pred[i * 8 + 2]: continue
        if y_true[i * 8 + 3] != y_pred[i * 8 + 3]: continue
        if y_true[i * 8 + 4] != y_pred[i * 8 + 4]: continue
        if y_true[i * 8 + 5] != y_pred[i * 8 + 5]: continue
        if y_true[i * 8 + 6] != y_pred[i * 8 + 6]: continue
        if y_true[i * 8 + 7] != y_pred[i * 8 + 7]: continue
        true_cases += 1
    aspect_strict_Acc = true_cases / total_cases

    return aspect_strict_Acc


def sentihood_macro_F1_t5(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of Sentihood.
    """
    p_all = 0
    r_all = 0
    count = 0
    for i in range(len(y_pred) // 8):
        a = set()
        b = set()
        for j in range(8):
            if y_pred[i * 8 + j] != 0:
                a.add(j)
            if y_true[i * 8 + j] != 0:
                b.add(j)
        if len(b) == 0: continue
        a_b = a.intersection(b)
        if len(a_b) > 0:
            p = len(a_b) / len(a)
            r = len(a_b) / len(b)
        else:
            p = 0
            r = 0
        count += 1
        p_all += p
        r_all += r
    Ma_p = p_all / count
    Ma_r = r_all / count
    aspect_Macro_F1 = 2 * Ma_p * Ma_r / (Ma_p + Ma_r)

    return Ma_p, Ma_r, aspect_Macro_F1


def sentihood_strict_acc(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of Sentihood.
    """
    total_cases=int(len(y_true)/4)
    true_cases=0
    for i in range(total_cases):
        if y_true[i*4]!=y_pred[i*4]:continue
        if y_true[i*4+1]!=y_pred[i*4+1]:continue
        if y_true[i*4+2]!=y_pred[i*4+2]:continue
        if y_true[i*4+3]!=y_pred[i*4+3]:continue
        true_cases+=1
    aspect_strict_Acc = true_cases/total_cases

    return aspect_strict_Acc


def sentihood_macro_F1(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of Sentihood.
    """
    p_all=0
    r_all=0
    count=0
    for i in range(len(y_pred)//4):
        a=set()
        b=set()
        for j in range(4):
            if y_pred[i*4+j]!=0:
                a.add(j)
            if y_true[i*4+j]!=0:
                b.add(j)
        if len(b)==0:continue
        a_b=a.intersection(b)
        if len(a_b)>0:
            p=len(a_b)/len(a)
            r=len(a_b)/len(b)
        else:
            p=0
            r=0
        count+=1
        p_all+=p
        r_all+=r
    Ma_p=p_all/count
    Ma_r=r_all/count
    aspect_Macro_F1 = 2*Ma_p*Ma_r/(Ma_p+Ma_r)

    return Ma_p, Ma_r, aspect_Macro_F1


def convert_pred_to_TAS_format(truth, preds):
    new_preds, trimmed_preds = [], []
    num_trimmed_sentences = 0

    for pred in preds:
        new_pred = []
        trim_flag = True
        for p in pred:
            # match = re.match(r"(The review expressed (\[([A-Za-z]+)\] opinion on \[(.+?)\] for \[(.+?)\](, )*)+)", p)
            if phr_sen == '':
                # match = re.match(r"(The review expressed (\[([A-Za-z]+)\] opinion on \[(.+?)\](, )*)+)", p)
                if eval_task == 'ASD':
                    match = re.match(r"(The review expressed (\[([A-Za-z]+)\] opinion on \[(.+?)\](, )*)+)", p)
                else:
                    match = re.match(r"(The review expressed (opinion on \[(.+?)\](, )*)+)", p)
            else:
                if eval_task == 'ASD':
                    match = re.match(r"(((LOCATION1|LOCATION2) - (.*)(\s)+(Positive|Negative)(\s)*)+)", p)
                else:
                    match = re.match(r"(((LOCATION1|LOCATION2) - (.*)(\s)*)+)", p)
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

    with open(f'{eval_task}{run}_{dir_prefix}_trimmed_preds1.txt', "w+") as f:
        f.write(f"Number of trimmed sentences={num_trimmed_sentences}\n\n")
        f.write("\n\n".join(trimmed_preds))

    preds = new_preds

    if not os.path.exists("predictions"):
        os.mkdir("predictions")

    # Saving the predictions if needed
    with open(f"predictions/{eval_task}_{dir_prefix}_predictions_{datetime.now()}.txt", "w") as f:
        for i, text in enumerate(df["input_text"].tolist()):
            f.write(str(text) + "\n\n")

            f.write("Truth:\n")
            f.write(truth[i] + "\n\n")

            f.write("Prediction:\n")
            for pred in preds[i]:
                f.write(str(pred) + "\n")
            f.write("________________________________________________________________________________\n")

    # exit(1)

    aspects = ["general", "price", "safety", "transit-location"]
    target1 = ["LOCATION1"]
    target2 = ["LOCATION2"]
    target1_aspects = [f"{tgt} - {asp}" for tgt in target1 for asp in aspects]
    target2_aspects = [f"{tgt} - {asp}" for tgt in target1 for asp in aspects]
    target1_aspect_map = {val: indx for indx, val in enumerate(target1_aspects)}
    target2_aspect_map = {val: indx for indx, val in enumerate(target2_aspects)}
    polarities = ["None", "Positive", "Negative"]
    polarity_map = {val: indx for indx, val in enumerate(polarities)}

    for pred_offset in range(3):
        # get the input text ids, and input text from the text_gen test set for this task
        # input_text_ids = df["input_text_ids"].tolist()
        input_text = df["input_text"].tolist()
        out_sent, out_true, out_pred = [], [], []
        for idx, inp_text in enumerate(input_text):
            # extract true and predicted aspect categories adn the polarities
            if phr_sen == '':
                true_aspects = re.findall(r"opinion on \[(.+?)\]", truth[idx])
                pred_aspects = re.findall(r"opinion on \[(.+?)\]", preds[idx][pred_offset])
            else:
                true_aspects = [each_op.split(" ~ ")[0] for each_op in truth[idx].split(" ~~ ")]
                pred_aspects = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  ")) if
                                op_idx % 2 == 0 and preds[idx][pred_offset] != '']

            location2 = False
            for each in true_aspects:
                if each in target2_aspect_map:
                    location2 = True

            length_of_target_aspects = (len(target2_aspects) + len(target1_aspects)) if location2 else len(target1_aspects)
            sent_out_true, sent_out_pred = [0] * length_of_target_aspects, [0] * length_of_target_aspects

            if eval_task == 'AD':
                for asp in true_aspects:
                    if asp in target1_aspect_map:
                        sent_out_true[target1_aspect_map[asp]] = 1
                    if location2 and (asp in target2_aspect_map):
                        sent_out_true[len(target1_aspects) + target2_aspect_map[asp]] = 1

                for asp in pred_aspects:
                    if asp in target1_aspect_map:
                        sent_out_pred[target1_aspect_map[asp]] = 1
                    if location2 and (asp in target2_aspect_map):
                        sent_out_pred[len(target1_aspects) + target2_aspect_map[asp]] = 1

            if eval_task == 'ASD':
                if phr_sen == '':
                    true_pol = re.findall(r" \[([A-Za-z]+)\] opinion on", truth[idx])
                    pred_pol = re.findall(r" \[([A-Za-z]+)\] opinion on", preds[idx][pred_offset])
                else:
                    true_pol = [each_op.split(" ~ ")[1] for each_op in truth[idx].split(" ~~ ")]
                    pred_pol = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  ")) if
                                op_idx % 2 == 1 and preds[idx][pred_offset] != '']

                assert len(true_pol) == len(true_aspects)
                assert len(pred_pol) == len(pred_aspects)

                for asp, pol in zip(true_aspects, true_pol):
                    if asp in target1_aspect_map and pol in polarity_map:
                        sent_out_true[target1_aspect_map[asp]] = polarity_map[pol]
                    if location2 and (asp in target2_aspect_map) and pol in polarity_map:
                        sent_out_true[len(target1_aspects) + target2_aspect_map[asp]] = polarity_map[pol]
                for asp, pol in zip(pred_aspects, pred_pol):
                    if asp in target1_aspect_map and pol in polarity_map:
                        sent_out_pred[target1_aspect_map[asp]] = polarity_map[pol]
                    if location2 and (asp in target2_aspect_map) and pol in polarity_map:
                        sent_out_pred[len(target1_aspects) + target2_aspect_map[asp]] = polarity_map[pol]

            out_true.extend(sent_out_true)
            out_pred.extend(sent_out_pred)
            out_sent.extend([inp_text] * length_of_target_aspects)

        print("\n\nlength of true and pred lists: \n\n", len(out_true), len(out_pred))

        prec, rec, f1 = sentihood_macro_F1(out_true, out_pred)
        print("*******************************************************\n\n")
        print(f"Aspect Strict Accuracy: {sentihood_strict_acc(out_true, out_pred)}")
        print(f"Aspect Macro Precision: {prec}\nAspect Macro Recall: {rec}\nAspect Macro F1: {f1}")
        if eval_task == 'ASD':
            sentiment_truth, sentiment_pred, sentiment_not_none_truth = [], [], []
            for t, p in zip(out_true, out_pred):
                if t != 0:
                    sentiment_truth.append(t - 1)
                    if p != 0:
                        sentiment_pred.append(p - 1)
                    else:
                        sentiment_pred.append(0)

            print(f"Sentiment Accuracy: {accuracy_score(sentiment_truth, sentiment_pred)}")
            print(classification_report(sentiment_truth, sentiment_pred))
            print(confusion_matrix(sentiment_truth, sentiment_pred))
        print("\n\n*******************************************************\n")

        out_df = pd.DataFrame(out_sent, columns=["sentence"])
        out_df["truth"] = out_true
        out_df["pred"] = out_pred
        out_df.to_csv(f"sentihood_output{pred_offset}.csv", index=False, header=True)


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
phr_sen = "_phrase" if sys.argv[3] == 'phrase' else ""
run = sys.argv[4]
print(f"dataset: {dataset}\ntask: {eval_task}\nphr_sen: {phr_sen}\nrun: {run}")
model_size = "base"
dir_prefix = f"{dataset}{phr_sen}"
df = pd.read_csv(f'data/{dataset}/test_{eval_task}{phr_sen}.csv')

tasks = df["prefix"].tolist()
analysis = False
# analysis = True

if not analysis:
    # Load the trained model
    # model = T5Model("t5", "outputs", args=model_args)
    model = T5Model("t5", f"results/{eval_task}{run}_{dir_prefix}/", args=model_args,
                    use_cuda=False if not torch.cuda.is_available() else True)

    # Prepare the data for testing
    to_predict = [
        prefix + ": " + str(input_text) for prefix, input_text in zip(df["prefix"].tolist(), df["input_text"].tolist())
    ]
    truth = df["target_text"].tolist()

    # Get the model predictions
    preds = model.predict(to_predict)

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
convert_pred_to_TAS_format(truth, preds)

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
#
# with open(f"results/result_{datetime.now()}.json", "w") as f:
#     json.dump(results_dict, f)
