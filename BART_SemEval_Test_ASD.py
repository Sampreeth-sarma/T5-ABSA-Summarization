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
from simpletransformers.config.model_args import Seq2SeqArgs
from simpletransformers.seq2seq import Seq2SeqModel
from simpletransformers.t5 import T5Model
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

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


def convert_pred_to_TAS_format(truth, preds, gold_xml_file=None):
    new_preds = []
    trimmed_preds = []
    num_trimmed_sentences = 0
    for pred in preds:
        new_pred = []
        trim_flag = True
        for p in [pred]:
            if phr_sen == '':
                if task == 'ASD':
                    match = re.match(
                        r"(The review expressed (\[(positive|negative|neutral)\] opinion on \[(.+?)\](, )*)+)", p)
                else:
                    match = re.match(
                        r"(The review expressed (opinion on \[(.+?)\](, )*)+)", p)
            else:
                if task == 'ASD':
                    match = re.match(r"((((f|F)ood|(d|D)rinks|(s|S)ervice|(a|A)mbience|(l|L)ocation|(r|R)estaurant"
                                     r")(\s)((g|G)eneral|(p|P)rices|(q|Q)uality|(s|S)tyle_options|(m|M)iscellaneous"
                                     r") ~ (positive|negative|neutral)( ~~ )?)+)", p)
                else:
                    match = re.match(r"((((f|F)ood|(d|D)rinks|(s|S)ervice|(a|A)mbience|(l|L)ocation|(r|R)estaurant"
                                     r")(\s)((g|G)eneral|(p|P)rices|(q|Q)uality|(s|S)tyle_options|(m|M)iscellaneous"
                                     r")( ~~ )?)+)", p)
            if match:
                out = match.groups()[0].strip().strip(",")
                new_pred.append(out)
            else:
                new_pred.append("")

            if task == 'ASD':
                opinion_patterns = re.findall(
                    r"(((f|F)ood|(d|D)rinks|(s|S)ervice|(a|A)mbience|(l|L)ocation|(r|R)estaurant"
                    r")(\s)?((g|G)eneral|(p|P)rices|(q|Q)uality|(s|S)tyle_options|(m|M)iscellaneous"
                    r")(\s)?~(\s)?(positive|negative|neutral))", p)
                if len(opinion_patterns) > 0:
                    # to reformat the opinion phrase into correct format based on correct opinions
                    opinion_patterns = " ~~ ".join(
                        [" ~ ".join([each_part.lower().strip() for each_part in each_op_pat[0].split("~")])
                         for each_op_pat in opinion_patterns])
                else:
                    opinion_patterns = ""
            else:
                if len(p.split("~~")) > 0:
                    op_parts = p.split("~~")
                    op_parts = [each_part.lower().strip() for each_part in op_parts if each_part.strip() != ""]
                    opinion_patterns = " ~~ ".join(op_parts)
                else:
                    opinion_patterns = ""

            if p != new_pred[-1] or p != opinion_patterns:
                if new_pred[-1] == "":
                    trimmed_preds.append(f"\n--------------")
                    if opinion_patterns != "":
                        new_pred[-1] = opinion_patterns
                else:
                    trimmed_preds.append(
                        f"Actual Prediction:\n\t{p}\nchanged pred: \n\t{new_pred[-1]}\nopinion patterns: \n\t{opinion_patterns}")
                    if new_pred[-1] != opinion_patterns:
                        new_pred[-1] = opinion_patterns
                if trim_flag:
                    trim_flag = False
                    num_trimmed_sentences += 1
        new_preds.append(new_pred[0])

        #     if p != new_pred[-1]:
        #         if new_pred[-1] == "":
        #             trimmed_preds.append("--------------")
        #         else:
        #             trimmed_preds.append(p + "\nchanged pred: \n" + new_pred[-1])
        #         if trim_flag:
        #             trim_flag = False
        #             num_trimmed_sentences += 1
        # new_preds.append(new_pred)

    with open(f'{task}{run}_{dir_prefix}/trimmed_preds.txt', "w+") as f:
        f.write(f"Number of trimmed sentences={num_trimmed_sentences}\n\n")
        f.write("\n\n".join(trimmed_preds))

    preds = new_preds

    if not os.path.exists("predictions"):
        os.mkdir("predictions")

    # Saving the predictions if needed
    with open(f"predictions/{task}_{dir_prefix}_predictions_{datetime.now()}.txt", "w") as f:
        for i, text in enumerate(df["input_text"].tolist()):
            f.write(str(text) + "\n\n")

            f.write("Truth:\n")
            f.write(truth[i] + "\n\n")

            f.write("Prediction:\n")
            f.write(str(preds[i]) + "\n")
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

    # get the input text ids, and input text from the text_gen test set for this task
    input_text = df["input_text"].tolist()
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
            pred_aspect = re.findall(r"opinion on \[(.+?)\]", preds[idx])
        else:
            true_aspect = [each_op.split(" ~ ")[0] for each_op in truth[idx].split(" ~~ ")]
            pred_aspect = [each_op.split(" ~ ")[0] for each_op in preds[idx].split(" ~~ ")]
            # pred_aspect = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  ")) if
            #                 op_idx % 2 == 0 and preds[idx][pred_offset] != '']

        true_aspect = ["#".join(each_asp.upper().split()) for each_asp in true_aspect]
        pred_aspect = ["#".join(each_asp.upper().split()) for each_asp in pred_aspect]

        if task == 'ASD':
            if phr_sen == '':
                true_pol = re.findall(r" \[([A-Za-z]+)\] opinion on", truth[idx])
                pred_pol = re.findall(r" \[([A-Za-z]+)\] opinion on", preds[idx])
            else:
                true_pol = [each_op.split(" ~ ")[1] for each_op in truth[idx].split(" ~~ ")]
                pred_pol = [each_op.split(" ~ ")[1] for each_op in preds[idx].split(" ~~ ") if preds[idx] != '' and each_op != '']
                # pred_pol = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  ")) if
                #                 op_idx % 2 == 1 and preds[idx][pred_offset] != '']

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

    if task == 'ASD':
        P = Common_Num / float(Pred_Num) if Pred_Num != 0 else 0
        R = Common_Num / float(True_Num)
        F = (2 * P * R) / float(P + R) if P != 0 else 0

        print('ASD task')
        print("\tP: ", P, "   R: ", R, "  F1: ", F)
        print('----------------------------------------------------\n\n')

    xml_string = ET.tostring(root)
    xml_write = DOM.parseString(xml_string)
    with open(f'evaluation_for_AD_TD_TAD/{task}{run}_{dir_prefix}_sentence0.xml', 'w') as handle:
        xml_write.writexml(handle, indent=' ', encoding='utf-8')
    print(f"\n\n\n*******\nGenarated target XML: {task}{run}_{dir_prefix}_sentence0.xml'\n*********\n\n")


if __name__ == '__main__':
    dataset = sys.argv[1]
    task = sys.argv[2]
    phr_sen = "_phrase" if sys.argv[3] == 'phrase' else ""
    run = sys.argv[4]
    model_size = sys.argv[5]

    print(f"dataset: {dataset}\ntask: {task}\nphr_sen: {phr_sen}\nrun: {run}\nmodel_size:{model_size}")

    use_crf = True
    # use_crf = False
    dir_prefix = f"{dataset}{phr_sen}{'_bart_' + model_size}{'_crf' if use_crf else ''}"

    model_args = Seq2SeqArgs()
    model_args.num_train_epochs = 100
    # model_args.no_save = True
    model_args.evaluate_generated_text = False
    model_args.evaluate_during_training = False
    model_args.evaluate_during_training_verbose = False
    model_args.output_dir = f"results/{task}{run}_{dir_prefix}/"
    model_args.save_steps = -1
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    # model_args.best_model_dir = f"{task}_{dir_prefix}/best_model/"
    model_args.max_length = 512
    model_args.max_seq_length = 512
    model_args.overwrite_output_dir = True
    # model_args.num_return_sequences = 3
    # model_args.top_k = 50,
    # model_args.top_p = 0.95,

    print(f"Reloading BART model from saved file in {task}{run}_{dir_prefix}")
    model_reloaded = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name=f"results/{task}{run}_{dir_prefix}/",
        args=model_args,
        use_cuda=False if not torch.cuda.is_available() else True
    )

    # Use the model for prediction

    print("\n\n".join(model_reloaded.predict(["The lemon chicken tasted like sticky sweet donuts and the honey walnut "
                                              "prawns , the few they actually give you . . . . . were not good .",
                                              "The wine list is incredible and extensive and diverse , the food is all "
                                              "incredible and the staff was all very nice , good at their jobs and "
                                              "cultured .",
                                              "I complained to the manager , but he was not even apologetic .",
                                              "Nice ambience , but highly overrated place .",
                                              "– Eggs , pancakes , potatoes , fresh fruit and yogurt - - everything they "
                                              "serve is delicious ."])))

    df = pd.read_csv(f'data/{dataset}/test_{task}{phr_sen}.csv')

    # tasks = df["prefix"].tolist()
    # analysis = False
    analysis = True

    if not analysis:
        # Load the trained model
        model = model_reloaded

        # Prepare the data for testing
        to_predict = [
            str(input_text) for prefix, input_text in zip(df["prefix"].tolist(), df["input_text"].tolist())
        ]
        truth = df["target_text"].tolist()

        # Get the model predictions
        preds = model.predict(to_predict)

        print(preds[0])

        with open(f'{task}{run}_{dir_prefix}/truth.pkl', "wb") as f:
            pickle.dump(truth, f)
        with open(f'{task}{run}_{dir_prefix}/preds.pkl', "wb") as f:
            pickle.dump(preds, f)

    else:
        with open(f'{task}{run}_{dir_prefix}/truth.pkl', "rb") as f:
            truth = pickle.load(f)
        with open(f'{task}{run}_{dir_prefix}/preds.pkl', "rb") as f:
            preds = pickle.load(f)

    # exit(0)

    convert_pred_to_TAS_format(truth, preds, f"evaluation_for_AD_TD_TAD/ABSA{15 if '15' in dataset else 16}_Restaurants_Test.xml")
