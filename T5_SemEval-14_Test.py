import os
import pickle
import re
import sys
import warnings
from datetime import datetime
from statistics import mean

import pandas as pd
import torch.cuda
from simpletransformers.t5 import T5Model
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

warnings.filterwarnings('ignore')


def f1(truths, preds):
    return mean([compute_f1(truth, pred) for truth, pred in zip(truths, preds)])


def exact(truths, preds):
    return mean([compute_exact(truth, pred) for truth, pred in zip(truths, preds)])


def semeval_PRF(y_true, y_pred):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2014.
    """
    s_all = 0
    g_all = 0
    s_g_all = 0
    for i in range(len(y_pred) // 5):
        s = set()
        g = set()
        for j in range(5):
            if y_pred[i * 5 + j] != 4:
                s.add(j)
            if y_true[i * 5 + j] != 4:
                g.add(j)
        if len(g) == 0: continue
        s_g = s.intersection(g)
        s_all += len(s)
        g_all += len(g)
        s_g_all += len(s_g)

    p = s_g_all / s_all
    r = s_g_all / g_all
    f = 2 * p * r / (p + r)

    return p, r, f


def semeval_Acc(y_true, y_pred, classes=4):
    """
    Calculate "Acc" of sentiment classification task of SemEval-2014.
    """
    assert classes in [2, 3, 4], "classes must be 2 or 3 or 4."

    if classes == 4:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] == 4: continue
            total += 1
            tmp = y_pred[i]
            if tmp == 4:
                continue
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total
    elif classes == 3:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] >= 3: continue
            total += 1
            tmp = y_pred[i]
            if tmp >= 3:
                continue
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total
    else:
        total = 0
        total_right = 0
        for i in range(len(y_true)):
            if y_true[i] >= 3 or y_true[i] == 1: continue
            total += 1
            tmp = y_pred[i]
            if tmp >= 3 or tmp == 1:
                continue
            if y_true[i] == tmp:
                total_right += 1
        sentiment_Acc = total_right / total

    return sentiment_Acc


def get_y_true(task_name):
    true_data_file = "data/semeval2014/test_NLI_B.csv"

    df = pd.read_csv(true_data_file, sep='\t', header=None).values
    y_true = []
    for i in range(len(df) // 5):
        if df[(i * 5)][1] == 1:
            n = 0
        elif df[(i * 5) + 1][1] == 1:
            n = 1
        elif df[(i * 5) + 2][1] == 1:
            n = 2
        elif df[(i * 5) + 3][1] == 1:
            n = 3
        else:
            n = 4

        y_true.append(n)

    return y_true


def convert_pred_to_TAS_format(truth, preds):
    new_preds = []
    for pred in preds:
        new_pred = []
        for p in pred:
            if phr_sen == '':
                # match = re.match(r"(The review expressed (\[([A-Za-z]+)\] opinion on \[(.+?)\](, )*)+)", p)
                if eval_task == 'ASD':
                    match = re.match(r"(The review expressed (\[([A-Za-z]+)\] opinion on \[(.+?)\](, )*)+)", p)
                else:
                    match = re.match(r"(The review expressed (opinion on \[(.+?)\](, )*)+)", p)
            else:
                if eval_task == 'ASD':
                    match = re.match(
                        r"(((food|service|ambiance|price|anecdotes)(\s)+(positive|negative|neutral|conflict)(\s)*)+)",
                        p)
                else:
                    match = re.match(r"(((food|service|ambience|price|anecdotes)(\s)*)+)", p)

            if match:
                out = match.groups()[0].strip().strip(",")
                new_pred.append(out)
            else:
                new_pred.append("")
        new_preds.append(new_pred)
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

    aspects = ["price", "anecdotes", "food", "ambiance", "service"]
    aspect_map = {val: indx for indx, val in enumerate(aspects)}
    if eval_task == 'ASD':
        polarities = ["positive", "neutral", "negative", "conflict", "none"]
        polarity_map = {val: indx for indx, val in enumerate(polarities)}

    for pred_offset in range(3):
        input_text = df["input_text"].tolist()
        out_sent, out_true, out_pred = [], [], []
        for idx, inp_text in enumerate(input_text):
            sent_out_true, sent_out_pred = [4] * len(aspects), [4] * len(aspects)
            # extract true and predicted aspect categories adn the polarities

            if phr_sen == '':
                true_aspects = re.findall(r"opinion on \[(.+?)\]", truth[idx])
                pred_aspects = re.findall(r"opinion on \[(.+?)\]", preds[idx][pred_offset])
            else:
                true_aspects = [each_op.split(" ~ ")[0] for each_op in truth[idx].split(" ~~ ")]
                pred_aspects = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  ")) if
                                op_idx % 2 == 0 and preds[idx][pred_offset] != '']
                # pred_pol = [[each_pol for each_pol in each_op.split(" ~ ")[2]] for each_op in preds[idx][pred_offset].split(" ~~ ")]

            if eval_task == 'AD':
                for asp in true_aspects:
                    if asp in aspect_map:
                        sent_out_true[aspect_map[asp]] = 1
                for asp in pred_aspects:
                    if asp in aspect_map:
                        sent_out_pred[aspect_map[asp]] = 1

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
                    if asp in aspect_map and pol in polarity_map:
                        sent_out_true[aspect_map[asp]] = polarity_map[pol]
                for asp, pol in zip(pred_aspects, pred_pol):
                    if asp in aspect_map and pol in polarity_map:
                        sent_out_pred[aspect_map[asp]] = polarity_map[pol]

            out_true.extend(sent_out_true)
            out_pred.extend(sent_out_pred)

        prec, rec, f1 = semeval_PRF(out_true, out_pred)
        print("*******************************************************\n\n")
        if eval_task == 'ASD':
            print(f"sentiment_Acc_4_classes: {semeval_Acc(out_true, out_pred, classes=4)}")
            print(f"sentiment_Acc_3_classes: {semeval_Acc(out_true, out_pred, classes=3)}")
            print(f"sentiment_Acc_2_classes: {semeval_Acc(out_true, out_pred, classes=2)}")
        print(f"Aspect Macro Precision: {prec}\nAspect Macro Recall: {rec}\nAspect Macro F1: {f1}")
        print("\n\n*******************************************************\n")

        out_df = pd.DataFrame(out_sent, columns=["sentence"])
        out_df["truth"] = out_true
        out_df["pred"] = out_pred
        out_df.to_csv(f"se-14_output{pred_offset}.csv", index=False, header=True)


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

# dataset = 'semeval-2014'
# eval_task = 'ASD'

dataset = sys.argv[1]
eval_task = sys.argv[2]
phr_sen = "_phrase" if sys.argv[3] == 'phrase' else ""
run = sys.argv[4]
print(f"dataset: {dataset}\ntask: {eval_task}\nphr_sen: {phr_sen}\nrun: {run}")

model_size = "base"
# model_size = "large"
# dir_prefix = f"{dataset}_{model_size}"

dir_prefix = f"{dataset}{phr_sen}"
df = pd.read_csv(f'data/{dataset}/test_{eval_task}{phr_sen}.csv')
print(df.head())

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

# with open(f"results/result_{datetime.now()}.json", "w") as f:
# json.dump(results_dict, f)
