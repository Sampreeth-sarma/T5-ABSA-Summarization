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

import collections
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


def convert_pred_to_TAS_format(truth, preds):
    new_preds = []
    trimmed_preds = []
    num_trimmed_sentences = 0
    for pred in preds:
        new_pred = []
        trim_flag = True
        for p in pred:
            if eval_task == 'ASD':
                match = re.match(r"(The review expressed (\[([A-Za-z]+)\] opinion on \[(.+?)\](, )*)+)", p)
            else:
                match = re.match(r"(The review expressed (opinion for \[(.+?)\](, )*)+)", p)
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
    with open(f"predictions/{eval_task}{run}_{dir_prefix}_predictions_{datetime.now()}.txt", "w") as f:
        for i, text in enumerate(df["input_text"].tolist()):
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
    gold_aspect_sentiment_list = gold_df["aspect_sentiment"].tolist()[:36]
    gold_aspect_sentiment_dict = {v: k for k, v in enumerate(gold_aspect_sentiment_list)}
    gold_yes_no = gold_df["yes_no"].tolist()
    gold_ner = gold_df["ner_tags"].tolist()

    for pred_offset in range(3):
        # get the input text ids, and input text from the text_gen test set for this task
        input_text_ids = df["input_text_ids"].tolist()
        input_text = df["input_text"].tolist()
        yes_no, yes_no_pred, text, true_ner, predict_ner = [], [], [], [], []
        wrong_count = 0
        dup_count = 0
        longest_prefix_count = 0
        for idx, inp_text in enumerate(input_text):
            wrong_flag = False
            # set the values of true yes_no values, true_ner, true_text of a sentence from the gold annotations
            # loaded earlier
            if eval_task == 'ASD':
                sentence_yes_no = gold_yes_no[idx * 36: (idx + 1) * 36]
            else:
                sentence_true_ner = gold_ner[idx * 36: (idx + 1) * 36]
                sentence_text = (['[CLS] ' + inp_text]) * 36

            # After running the regex, if the prediction string is empty, then, directly assign the default values to
            # the prediction yes_no, prediction ner
            sent_ner_len = len(inp_text.split())

            if eval_task == 'ASD':
                sentence_yes_no_pred = [0] * 36
            else:
                sentence_predict_ner = [" ".join(['O'] * sent_ner_len)] * 36

            if preds[idx][pred_offset] != "":
                # check if the true target sentence and the predicted target sentence are same
                # if truth[idx] == preds[idx][pred_offset]:
                #     # if same, we can directly assign the gold values as the predicted values
                #     sentence_yes_no_pred = gold_yes_no[idx * 36: (idx + 1) * 36]
                #     assert collections.Counter(sentence_yes_no_pred) == collections.Counter(sentence_yes_no)
                #
                #     sentence_predict_ner = gold_ner[idx * 36: (idx + 1) * 36]
                #     assert collections.Counter(sentence_predict_ner) == collections.Counter(sentence_true_ner)
                #
                # else:
                # if not same, we continue to extract the labels from teh predicted text and assign them
                # first we assign default values to the yes_no_pred, and ner_tags_pred

                if eval_task == 'ASD':
                    assert len(sentence_yes_no_pred) == len(sentence_yes_no)
                else:
                    assert len(sentence_predict_ner) == len(sentence_true_ner)
                    assert len(sentence_predict_ner[0]) == len(sentence_true_ner[0])

                # extract true and predicted aspect categories adn the polarities
                if eval_task == 'ASD':

                    true_aspects = re.findall(r"opinion on \[(.+?)\]", truth[idx])
                    pred_aspects = re.findall(r"opinion on \[(.+?)\]", preds[idx][pred_offset])
                    true_pol = re.findall(r" \[([A-Za-z]+)\] opinion on", truth[idx])
                    pred_pol = re.findall(r" \[([A-Za-z]+)\] opinion on", preds[idx][pred_offset])

                    assert len(true_pol) == len(true_aspects)
                    assert len(pred_pol) == len(pred_aspects)

                    # combine the aspect categories and the polarities to form true and predicted aspect-sentiment
                    # strings
                    true_aspect_pol = [true_aspects[i] + " " + true_pol[i] for i in range(len(true_pol))]
                    pred_aspect_pol = [pred_aspects[i] + " " + pred_pol[i] for i in range(len(pred_pol))]

                    # find the indexes of the aspect categories based on the dict of 36 values
                    # This can be used to select the particular TAS tuple out of the 36 possibilities of a sentence
                    true_aspect_pol_idx = [gold_aspect_sentiment_dict[each] for each in true_aspect_pol if
                                           each in gold_aspect_sentiment_dict]
                    pred_aspect_pol_idx = [gold_aspect_sentiment_dict[each] for each in pred_aspect_pol if
                                       each in gold_aspect_sentiment_dict]

                    if len(pred_aspect_pol_idx) > 0:
                        # similarly, get the indexes of the gold aspect categories of the sentence
                        # this is used to verify whether the true_indices == gold_indices for aspect-sentiments
                        gold_yes_no_idx = [i for i, val in enumerate(gold_yes_no[idx * 36: (idx + 1) * 36]) if val == 1]
                        assert collections.Counter(set(gold_yes_no_idx)) == collections.Counter(
                            set(true_aspect_pol_idx))
                        for each_asp_idx in pred_aspect_pol_idx:
                            sentence_yes_no_pred[each_asp_idx] = 1
                else:
                    # extract true and predicted targets and their indexes for the respective aspect-sentiment pair
                    true_target = re.findall(r" for \[(.+?)\]", truth[idx])
                    pred_target = re.findall(r" for \[(.+?)\]", preds[idx][pred_offset])

                    true_target_idx = []
                    for each_target in true_target:
                        if each_target != 'NULL':
                            sub_idx = getsubidx(inp_text.split(), each_target.split())
                            if inp_text.count(each_target) > 1:
                                dup_count += 1
                                # print(f"{dup_count}: Target: {each_target}\nText: {inp_text}\n\n")
                            if sub_idx != -1:
                                true_target_idx.append(
                                    [it for it in range(sub_idx, (sub_idx + len(each_target.split())))])
                            else:
                                true_target_idx.append([])
                        else:
                            true_target_idx.append([])

                    # exit(1)

                    pred_target_idx = []
                    for each_target in pred_target:
                        if each_target != 'NULL':

                            # clean the target word before finding it's index
                            # The intuition is changing the word "Ray' s" ----> "Ray ' s"
                            tgt = clean_str(each_target)
                            if each_target != tgt:
                                print(f"changing '{each_target}' to  '{tgt}'\n")
                                each_target = tgt

                            # match the longest prefix from the sentence for each target word and replace the word
                            # with the one from the sentence if there >80% match compared to the target word
                            # else don't change

                            # new_target_str = ""
                            # for each_target_word in each_target.split():
                            #     if each_target_word not in inp_text.split():
                            #         new_each_target_word = []
                            #         for each_inp_word in inp_text.split():
                            #             if (len(longestCommonPrefix(
                            #                     [each_inp_word, each_target_word])) / len(each_target_word)) > 0.8:
                            #                 new_each_target_word.append(each_inp_word)
                            #         if len(new_each_target_word) == 0:
                            #             new_target_str += f" {each_target_word}"
                            #         else:
                            #             new_target_str += " ".join(new_each_target_word)
                            #     else:
                            #         new_target_str += f" {each_target_word}"
                            # new_target_str = new_target_str.strip()
                            # if new_target_str != each_target:
                            #     longest_prefix_count += 1
                            #     print(f"{longest_prefix_count} Longest Prefix Match Changes - {each_target}: {new_target_str}\n{inp_text}\n")
                            #     each_target = new_target_str

                            # Find the indices of the target expression in the sentence
                            sub_idx = getsubidx(inp_text.split(), each_target.split())
                            if sub_idx != -1:
                                pred_target_idx.append(
                                    [it for it in range(sub_idx, (sub_idx + len(each_target.split())))])
                            else:
                                pred_target_idx.append([])
                        else:
                            pred_target_idx.append([])

                    tgt_count = 0
                    for each_tgt_idx in pred_target_idx:
                        # *************************** changing each_asp_idx to tgt_count ******************************
                        tgt_ner_loc = sentence_predict_ner[tgt_count].split()
                        for each_idx in each_tgt_idx:
                            tgt_ner_loc[each_idx] = 'T'
                        sentence_predict_ner[tgt_count] = " ".join(tgt_ner_loc).strip()
                        tgt_count += 3

            if eval_task != 'ASD':
                sentence_predict_ner = ['[CLS] ' + each for each in sentence_predict_ner]
                sentence_true_ner = ['[CLS] ' + each for each in sentence_true_ner]

            if eval_task == 'ASD':
                assert len(sentence_yes_no) == len(sentence_yes_no_pred)
                yes_no.extend(sentence_yes_no)
                yes_no_pred.extend(sentence_yes_no_pred)
            else:
                assert len(sentence_text) == len(sentence_true_ner)
                assert len(sentence_predict_ner) == len(sentence_true_ner)
                text.extend(sentence_text)
                true_ner.extend(sentence_true_ner)
                predict_ner.extend(sentence_predict_ner)
        print(wrong_count)

        if eval_task == 'ASD':
            out_df = pd.DataFrame(yes_no, columns=['yes_not'])
            out_df['yes_not_pred'] = yes_no_pred
        else:
            out_df = pd.DataFrame(text, columns=['sentence'])
            out_df['true_ner'] = true_ner
            out_df['predict_ner'] = predict_ner

        out_df.to_csv(f"{eval_task}{run}_{dir_prefix}/converted_predictions{pred_offset}.txt",
                      sep="\t", index=False, header=True)


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
    model = T5Model("t5", f"{eval_task}{run}_{dir_prefix}/", args=model_args,
                    use_cuda=False if not torch.cuda.is_available() else True)

    # Prepare the data for testing
    to_predict = [
        prefix + ": " + str(input_text) for prefix, input_text in zip(df["prefix"].tolist(), df["input_text"].tolist())
    ]
    truth = df["target_text"].tolist()

    # Get the model predictions
    preds = model.predict(to_predict)

    print("\n".join(preds[0]))

    with open('TASD1_truth.pkl', "wb") as f:
        pickle.dump(truth, f)
    with open('TASD1_preds.pkl', "wb") as f:
        pickle.dump(preds, f)

else:
    with open('TASD1_truth.pkl', "rb") as f:
        truth = pickle.load(f)
    with open('TASD1_preds.pkl', "rb") as f:
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
