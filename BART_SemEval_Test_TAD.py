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


def convert_pred_to_TAS_format(truth, preds, gold_xml_file=None):
    new_preds = []
    trimmed_preds = []
    num_trimmed_sentences = 0
    for pred in preds:
        new_pred = []
        trim_flag = True
        for p in [pred]:
            if phr_sen == '':
                match = re.match(r"(The review expressed (opinion on \[(.+?)\] for \[(.+?)\](, )*)+)", p)
            else:
                match = re.match(
                    r"(([\sA-Za-z0-9(),!?.\$\*\+;/:@&#%\"=\-'`–’é]+ ~ ((food|drinks|service|ambience|location|restaurant)(\s)+(general|prices|quality|style_options|miscellaneous))(\s)*)+)",
                    p)
            if match:
                out = match.groups()[0].strip().strip(",")
                new_pred.append(out)
            else:
                new_pred.append("")

            opinion_patterns = re.findall(
                r"([\sA-Za-z0-9(),!?.\$\*\+;/:@&#%\"=\-'`–’é]+(\s)?~(\s)?(((f|F)ood|(d|D)rinks|(s|S)ervice|(a|A)mbience|(l|L)ocation|(r|R)estaurant)(\s)?("
                r"(g|G)eneral|(p|P)rices|(q|Q)uality|(s|S)tyle_options|(m|M)iscellaneous)))",
                p)
            if len(opinion_patterns) > 0:
                # to reformat the opinion phrase into correct format based on correct opinions
                opinion_patterns = " ~~ ".join(
                    [" ~ ".join([each_part.strip() for each_part in each_op_pat[0].split("~")]) for each_op_pat in
                     opinion_patterns])
                new_opinion_pattern = []
                for each_op_pat in opinion_patterns.split(" ~~ "):
                    op_aspect = each_op_pat.split(" ~ ")[1].lower()
                    if len(op_aspect.split()) == 1:
                        # to reformat the aspect category if there is a missing space or inconsistent capitalization
                        op_aspect = " ".join([each.strip() for each in
                                              re.split(r"(food|drinks|service|ambience|location|restaurant)",
                                                       op_aspect)
                                              if each.strip() != ''])
                    each_op_part = each_op_pat.split(" ~ ")
                    each_op_part[1] = op_aspect
                    new_opinion_pattern.append(" ~ ".join(each_op_part))
                opinion_patterns = " ~~ ".join(new_opinion_pattern)
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

    with open(f'{task}{run}_{dir_prefix}/trimmed_preds1.txt', "w+") as f:
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
            pred_aspects = re.findall(r"opinion on \[(.+?)\]", preds[idx])
        else:
            pred_aspects = [each_op.split(" ~ ")[1] for each_op in preds[idx].split(" ~~ ")]
            # pred_aspects = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  ")) if
            #                 op_idx % 2 == 1 and preds[idx][pred_offset] != '']

        # extract true and predicted targets and their indexes for the respective aspect-sentiment pair
        if phr_sen == '':
            pred_target = re.findall(r"\] for \[(.+?)\]", preds[idx])
        else:
            pred_target = [each_op.split(" ~ ")[0] for each_op in preds[idx].split(" ~~ ")]
            # pred_target = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  ")) if
            #                 op_idx % 2 == 0 and preds[idx][pred_offset] != '']

        # If any aspect polarity is dropped by any chance, then, we have to exclude that respective
        # target also
        if len(pred_aspects) != len(pred_target):
            pred_target = pred_target[:len(pred_aspects)]

        pred_target_idx = []
        for each_target in pred_target:
            if each_target != 'NULL':

                # clean the target word before finding it's index
                # The intuition is changing the word "Ray' s" ----> "Ray ' s"
                tgt = clean_str(each_target)
                if each_target != tgt:
                    # print(f"changing '{each_target}' to  '{tgt}'\n")
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
                #     # print(f"{longest_prefix_count} Longest Prefix Match Changes - {each_target}: {new_target_str}\n{inp_text}\n")
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

        # verify if number of polarities == number of targets
        assert len(pred_aspects) == len(pred_target_idx)

        gold_sentence = inp_text.split()
        xml_sentence = current_sen.find('text').text

        for each_asp, each_tgt_idx in zip(pred_aspects, pred_target_idx):
            if len(each_tgt_idx) == 0:
                op = ET.Element('Opinion')
                op.set('target', 'NULL')
                op.set('category', "#".join(each_asp.upper().split()))
                op.set('polarity', "")
                op.set('from', '0')
                op.set('to', '0')
                current_opinions.append(op)
            else:
                # for x in pred_target_idx:
                start = each_tgt_idx[0]
                end = len(each_tgt_idx) + start
                target_sub_seq = gold_sentence[start: end]
                while '(' in target_sub_seq:
                    target_sub_seq[target_sub_seq.index('(')] = '\('
                while ')' in target_sub_seq:
                    target_sub_seq[target_sub_seq.index(')')] = '\)'
                while '$' in target_sub_seq:
                    target_sub_seq[target_sub_seq.index('$')] = '\$'
                target_match = re.compile('\\s*'.join(target_sub_seq))
                # target_match = re.compile('\\s*'.join(sentence[start:end]))
                sentence_org = ' '.join(gold_sentence)
                target_match_list = re.finditer(target_match, sentence_org)
                true_idx = 0
                for m in target_match_list:
                    if start == sentence_org[0:m.start()].count(' '):
                        break
                    true_idx += 1

                target_match_list = re.finditer(target_match, xml_sentence)
                match_list = []
                for m in target_match_list:
                    match_list.append(str(m.start()) + '###' + str(len(m.group())) + '###' + m.group())
                if len(match_list) < true_idx + 1:
                    print("Error!!!!!!!!!!!!!!!!!!!!!")
                    print(len(match_list))
                    print(target_match)
                    print(sentence_org)
                else:
                    info_list = match_list[true_idx].split('###')
                    target = info_list[2]
                    from_idx = info_list[0]
                    to_idx = str(int(from_idx) + int(info_list[1]))
                    op = ET.Element('Opinion')
                    op.set('target', target)
                    op.set('category', "#".join(each_asp.upper().split()))
                    op.set('polarity', "")
                    op.set('from', from_idx)
                    op.set('to', to_idx)
                    current_opinions.append(op)

    xml_string = ET.tostring(root)
    xml_write = DOM.parseString(xml_string)
    with open(f'evaluation_for_AD_TD_TAD/{task}{run}_{dir_prefix}_sentence0.xml', 'w') as handle:
        xml_write.writexml(handle, indent=' ', encoding='utf-8')
    print(f"\n\n\n*******\n{task}{run}_{dir_prefix}_sentence0.xml'\n*********\n\n")


if __name__ == '__main__':

    dataset = sys.argv[1]
    task = sys.argv[2]
    phr_sen = "_phrase" if sys.argv[3] == 'phrase' else ""
    run = sys.argv[4]
    model_size = sys.argv[5]

    print(f"dataset: {dataset}\ntask: {task}\nphr_sen: {phr_sen}\nrun: {run}\nmodel_size:{model_size}")

    train = False
    # train = True
    dir_prefix = f"{dataset}{phr_sen}{'_bart_' + model_size}"

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
