import collections
import logging
import math
import os
import pickle
import re
import sys
import warnings
from datetime import datetime
from multiprocessing import Pool

import pandas as pd
import simpletransformers.seq2seq.seq2seq_utils
import torch.cuda
from dataclasses import asdict
from torch import nn
from torchcrf import CRF

from BART_SemEval_Test_ASD import convert_pred_to_TAS_format as ASD
from BART_SemEval_Test_TAD import convert_pred_to_TAS_format as TAD
from BART_SemEval_Test_TSD import convert_pred_to_TAS_format as TSD

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, RandomSampler, DataLoader
from tqdm.auto import tqdm, trange
from simpletransformers.seq2seq.seq2seq_utils import (
    Seq2SeqDataset,
    load_hf_dataset,
)
from transformers import AdamW, Adafactor, get_constant_schedule, get_constant_schedule_with_warmup, \
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


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
    string = re.sub(r"’", " \’ ", string)
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
    match_count = 0
    for pred, gold in zip(preds, truth):
        new_pred = []
        trim_flag = True
        for p in [pred]:
            match = re.match(
                r"(([\sA-Za-z0-9(),!?.\$\*\+;/:@&#%\"=\-'`–’é]+ ~ ((food|drinks|service|ambience|location|restaurant)"
                r"(\s)(general|prices|quality|style_options|miscellaneous)) ~ (positive|negative|neutral)( ~~ )?)+)",
                p)
            if match:
                match_count += 1
                out = match.groups()[0].strip().strip("~~").strip()
                new_pred.append(out)
            else:
                new_pred.append("")

            opinion_patterns = re.findall(
                r"([\sA-Za-z0-9(),!?.\$\*\+;/:@&#%\"=\-'`–’é]+(\s)?~(\s)?(((f|F)ood|(d|D)rinks|(s|S)ervice|(a|A)mbience|(l|L)ocation|(r|R)estaurant)(\s)?("
                r"(g|G)eneral|(p|P)rices|(q|Q)uality|(s|S)tyle_options|(m|M)iscellaneous))(\s)?~(\s)?(positive|negative|neutral))",
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
                    trimmed_preds.append(f"Truth:\n{gold}\n--------------")
                    # if opinion_patterns != "":
                    #     new_pred[-1] = opinion_patterns
                else:
                    trimmed_preds.append(
                        f"Truth:\n\t{gold}\nActual Prediction:\n\t{p}\nchanged pred: \n\t{new_pred[-1]}\nopinion patterns: \n\t{opinion_patterns}")
                    # if new_pred[-1] != opinion_patterns:
                    #     new_pred[-1] = opinion_patterns
                if trim_flag:
                    trim_flag = False
                    num_trimmed_sentences += 1
        new_preds.append(new_pred[0])

    with open(f'results/{task}{run}_{dir_prefix}/trimmed_preds.txt', "w+") as f:
        f.write(f"Number of trimmed sentences={num_trimmed_sentences}\n\n")
        f.write("\n\n".join(trimmed_preds))

    # exit(0)

    preds = new_preds

    # Saving the predictions if needed
    with open(f"predictions/{task}{run}_{dir_prefix}_predictions_{datetime.now()}.txt", "w") as f:
        for i, text in enumerate(df["input_text"].tolist()):
            f.write(str(text) + "\n\n")

            f.write("Truth:\n")
            f.write(truth[i] + "\n\n")

            f.write("Prediction:\n")
            f.write(str(preds[i]) + "\n")
            f.write("________________________________________________________________________________\n")

    # print(match_count)
    # exit(1)

    def getsubidx(x, y):
        l1, l2 = len(x), len(y)
        for i in range(l1):
            if x[i:i + l2] == y:
                return i
        return -1

    num_of_comb = 36 if dataset == 'semeval-2016' else 39

    # get the gold annotations for the aspect-sentiment, yes_no, ner_tags from the TAS-BERT test file
    gold_df = pd.read_csv(f'data/{dataset}/test_TAS.tsv', sep="\t")
    gold_aspect_sentiment_list = gold_df["aspect_sentiment"].tolist()[:num_of_comb]
    gold_aspect_sentiment_dict = {v: k for k, v in enumerate(gold_aspect_sentiment_list)}
    gold_yes_no = gold_df["yes_no"].tolist()
    gold_ner = gold_df["ner_tags"].tolist()

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
        sentence_yes_no = gold_yes_no[idx * num_of_comb: (idx + 1) * num_of_comb]
        sentence_true_ner = gold_ner[idx * num_of_comb: (idx + 1) * num_of_comb]
        sentence_text = (['[CLS] ' + inp_text]) * num_of_comb

        # After running the regex, if the prediction string is empty, then, directly assign the default values to
        # the prediction yes_no, prediction ner
        sent_ner_len = len(inp_text.split())

        sentence_yes_no_pred = [0] * num_of_comb
        sentence_predict_ner = [" ".join(['O'] * sent_ner_len)] * num_of_comb
        if preds[idx] != "":

            assert len(sentence_predict_ner) == len(sentence_true_ner)
            assert len(sentence_predict_ner[0]) == len(sentence_true_ner[0])
            assert len(sentence_yes_no_pred) == len(sentence_yes_no)

            true_aspects = [each_op.split(" ~ ")[1] for each_op in truth[idx].split(" ~~ ")]
            true_pol = [each_op.split(" ~ ")[2] for each_op in truth[idx].split(" ~~ ")]
            pred_aspects = [each_op.split(" ~ ")[1] for each_op in preds[idx].split(" ~~ ")]
            pred_pol = [each_op.split(" ~ ")[2] for each_op in preds[idx].split(" ~~ ")]
            # pred_aspects = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  ")) if
            #                 op_idx % 3 == 1]
            # pred_pol = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  ")) if
            #             op_idx % 3 == 2]
            # pred_pol = [[each_pol for each_pol in each_op.split(" ~ ")[2]] for each_op in preds[idx][pred_offset].split(" ~~ ")]

            assert len(true_pol) == len(true_aspects)
            assert len(pred_pol) == len(pred_aspects)

            # combine the aspect categories and the polarities to form true and predicted aspect-sentiment
            # strings
            true_aspect_pol = [true_aspects[i] + " " + true_pol[i] for i in range(len(true_pol))]
            pred_aspect_pol = [pred_aspects[i] + " " + pred_pol[i] for i in range(len(pred_pol))]

            # find the indexes of the aspect categories based on the dict of num_of_comb values
            # This can be used to select the particular TAS tuple out of the num_of_comb possibilities of a sentence
            true_aspect_pol_idx = [gold_aspect_sentiment_dict[each] for each in true_aspect_pol if
                                   each in gold_aspect_sentiment_dict]
            pred_aspect_pol_idx = [gold_aspect_sentiment_dict[each] for each in pred_aspect_pol if
                                   each in gold_aspect_sentiment_dict]

            if len(pred_aspect_pol_idx) > 0:
                # similarly, get the indexes of the gold aspect categories of the sentence
                # this is used to verify whether the true_indices == gold_indices for aspect-sentiments
                gold_yes_no_idx = [i for i, val in
                                   enumerate(gold_yes_no[idx * num_of_comb: (idx + 1) * num_of_comb]) if val == 1]
                assert collections.Counter(set(gold_yes_no_idx)) == collections.Counter(
                    set(true_aspect_pol_idx))

                true_target = [each_op.split(" ~ ")[0] for each_op in truth[idx].split(" ~~ ")]
                pred_target = [each_op.split(" ~ ")[0] for each_op in preds[idx].split(" ~~ ")]
                # pred_target = [tgt_asp_pol for op_idx, tgt_asp_pol in enumerate(preds[idx][pred_offset].split("  "))
                #                if
                #                op_idx % 3 == 0]

                # if len(pred_target) > 1:
                #     print(pred_target)

                # If any aspect polarity is dropped by any chance, then, we have to exclude that respective
                # target also
                if len(pred_aspect_pol_idx) != len(pred_target):
                    pred_target = pred_target[:len(pred_aspect_pol_idx)]

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

                # verify if number of polarities == number of targets
                assert len(pred_aspect_pol_idx) == len(pred_target_idx)

                # find the gold target indexes to verify the correctness of true_target_idx
                gold_target_idx = [sorted(
                    [each_ner_tag_idx for each_ner_tag_idx, each_ner_tag in
                     enumerate(sentence_true_ner[each_idx].split())
                     if
                     each_ner_tag != 'O']) for each_idx in gold_yes_no_idx]
                assert len([item for sublist in gold_target_idx for item in sublist]) == \
                       len([item for sublist in true_target_idx for item in sublist])
                # assert collections.Counter([item for sublist in gold_target_idx for item in sublist]) == \
                #        collections.Counter([item for sublist in true_target_idx for item in sublist])

                for each_asp_idx, each_tgt_idx in zip(pred_aspect_pol_idx, pred_target_idx):
                    sentence_yes_no_pred[each_asp_idx] = 1
                    # if true_aspect_pol_idx == pred_aspect_pol_idx and true_target_idx == pred_target_idx:
                    #     sentence_predict_ner = sentence_true_ner
                    #     if not wrong_flag:
                    #         wrong_flag = True
                    #         # print(input_text[idx])
                    #         if len(pred_target_idx) == 1 and len(pred_target_idx[0]) == 0:
                    #             wrong_count += 1
                    # else:
                    tgt_ner_loc = sentence_predict_ner[each_asp_idx].split()
                    for each_idx in each_tgt_idx:
                        tgt_ner_loc[each_idx] = 'T'
                    sentence_predict_ner[each_asp_idx] = " ".join(tgt_ner_loc).strip()

        sentence_predict_ner = ['[CLS] ' + each for each in sentence_predict_ner]
        sentence_true_ner = ['[CLS] ' + each for each in sentence_true_ner]

        assert len(sentence_yes_no) == len(sentence_yes_no_pred)
        assert len(sentence_yes_no_pred) == len(sentence_text)
        assert len(sentence_text) == len(sentence_true_ner)
        assert len(sentence_predict_ner) == len(sentence_true_ner)

        yes_no.extend(sentence_yes_no)
        yes_no_pred.extend(sentence_yes_no_pred)
        text.extend(sentence_text)
        true_ner.extend(sentence_true_ner)
        predict_ner.extend(sentence_predict_ner)
    # print(wrong_count)

    out_df = pd.DataFrame(yes_no, columns=['yes_not'])
    out_df['yes_not_pred'] = yes_no_pred
    out_df['sentence'] = text
    out_df['true_ner'] = true_ner
    out_df['predict_ner'] = predict_ner

    out_df.to_csv(f"results/{task}{run}_{dir_prefix}/converted_predictions0.txt",
                  sep="\t", index=False, header=True)


def construct_ner_tags(target_texts, ner_tags):
    for tgt_txt in target_texts:
        sentence_ner_tags = []
        for op_idx, each_op in enumerate(tgt_txt.split(" ~~ ")):
            for part_idx, each_part in enumerate(each_op.split(" ~ ")):
                if task == "TASD":
                    for tok_idx, each_token in enumerate(each_part.split()):
                        if part_idx == 0:
                            if tok_idx == 0:
                                sentence_ner_tags.append("B-tgt")
                            else:
                                sentence_ner_tags.append("I-tgt")
                        elif part_idx == 1:
                            if tok_idx == 0:
                                sentence_ner_tags.append("B-asp")
                            else:
                                sentence_ner_tags.append("I-asp")
                        else:
                            sentence_ner_tags.append("B-pol")
                    if part_idx != 2:
                        sentence_ner_tags.append("B-sep")
                elif task == "ASD":
                    for tok_idx, each_token in enumerate(each_part.split()):
                        if part_idx == 0:
                            if tok_idx == 0:
                                sentence_ner_tags.append("B-asp")
                            else:
                                sentence_ner_tags.append("I-asp")
                        else:
                            sentence_ner_tags.append("B-pol")
                    if part_idx != 1:
                        sentence_ner_tags.append("B-sep")
                elif task == "AD":
                    for tok_idx, each_token in enumerate(each_part.split()):
                        if tok_idx == 0:
                            sentence_ner_tags.append("B-asp")
                        else:
                            sentence_ner_tags.append("I-asp")
                elif task == "TD":
                    for tok_idx, each_token in enumerate(each_part.split()):
                        if tok_idx == 0:
                            sentence_ner_tags.append("B-tgt")
                        else:
                            sentence_ner_tags.append("I-tgt")
                elif task == "TSD":
                    for tok_idx, each_token in enumerate(each_part.split()):
                        if part_idx == 0:
                            if tok_idx == 0:
                                sentence_ner_tags.append("B-tgt")
                            else:
                                sentence_ner_tags.append("I-tgt")
                        else:
                            sentence_ner_tags.append("B-pol")
                    if part_idx != 1:
                        sentence_ner_tags.append("B-sep")
                elif task == "TAD":
                    for tok_idx, each_token in enumerate(each_part.split()):
                        if part_idx == 0:
                            if tok_idx == 0:
                                sentence_ner_tags.append("B-tgt")
                            else:
                                sentence_ner_tags.append("I-tgt")
                        elif part_idx == 1:
                            if tok_idx == 0:
                                sentence_ner_tags.append("B-asp")
                            else:
                                sentence_ner_tags.append("I-asp")
                    if part_idx != 1:
                        sentence_ner_tags.append("B-sep")
            if op_idx < (len(tgt_txt.split(" ~~ ")) - 1):
                sentence_ner_tags.append("B-opsep")
        ner_tags.append(" ".join(sentence_ner_tags))

    return ner_tags


def reconstruct_ner_tags_for_encoded_targets(target_text, ner_tags, tokenizer, args):
    # text = "staff ~ service general ~ positive ~~ restaurant ~ restaurant general ~ negative"
    # ner_tags = "B-tgt I-tgt I-tgt B-sep B-asp I-asp B-sep B-pol B-opsep B-tgt I-tgt I-tgt I-tgt I-tgt B-sep B-asp I-asp B-sep B-pol"
    # encoded_text = tokenizer.encode(text)[1:-1]
    text = target_text
    encoded_text_org = \
        tokenizer.batch_encode_plus([text], max_length=args.max_seq_length, padding="max_length", return_tensors="pt",
                                    truncation=True)["input_ids"].squeeze().tolist()
    encoded_text = encoded_text_org[1:]  # to get rid of the start token
    enc_i = 0
    i = 0
    # new_ner_tags = ["[START]"]
    new_ner_tags = []
    while i < len(text.split()):
        if tokenizer.decode([encoded_text[enc_i]]).lstrip(' ') == text.split()[i]:
            # print(text.split()[i])
            new_ner_tags.append(ner_tags.split()[i])
            enc_i += 1
        else:
            enc_str = ""
            pos = 0
            while enc_str != text.split()[i]:
                enc_str = tokenizer.decode(encoded_text[enc_i: (enc_i + pos + 1)]).lstrip(' ')
                if pos == 0:
                    new_ner_tags.append(ner_tags.split()[i])
                else:
                    if "B-" in ner_tags.split()[i]:
                        tg = "I-" + ner_tags.split()[i].split("B-")[-1]
                    else:
                        tg = ner_tags.split()[i]
                    new_ner_tags.append(tg)
                pos += 1
            enc_i += pos
            # print(text.split()[i])
        i += 1
    # print(enc_i - i)
    new_ner_tags.append("[END]")
    new_ner_tags.extend(["[PAD]"] * (args.max_seq_length - (enc_i + 2)))

    text_to_tag = []
    for e_text, tag in zip(encoded_text_org, new_ner_tags):
        text_to_tag.append(f"{tokenizer.decode([e_text]).lstrip(' ')} : {tag2idx[tag]}")

    # print("\n".join(text_to_tag))
    # print(len(text_to_tag))

    return torch.LongTensor([tag2idx[each] for each in new_ner_tags])


def preprocess_data_bart(data):
    input_text, target_text, ner_tags, tokenizer, args = data

    input_ids = tokenizer.batch_encode_plus(
        [input_text], max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
    )

    target_ids = tokenizer.batch_encode_plus(
        [target_text], max_length=args.max_seq_length, padding="max_length", return_tensors="pt", truncation=True
    )

    target_ner_tags = reconstruct_ner_tags_for_encoded_targets(target_text, ner_tags, tokenizer, args)
    target_ner_mask = (target_ner_tags != 1).float()
    return {
        "source_ids": input_ids["input_ids"].squeeze(),
        "source_mask": input_ids["attention_mask"].squeeze(),
        "target_ids": target_ids["input_ids"].squeeze(),
        "target_ner_tags": target_ner_tags,
        "target_ner_mask": target_ner_mask,
    }


class SimpleSummarizationDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        self.tokenizer = tokenizer

        cached_features_file = os.path.join(
            args.cache_dir, args.model_name + "_cached_" + str(args.max_seq_length) + str(len(data))
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            data = [
                (input_text, target_text, ner_tags, tokenizer, args)
                for input_text, target_text, ner_tags in zip(data["input_text"], data["target_text"], data["ner_tags"])
            ]

            preprocess_fn = preprocess_data_bart

            if (mode == "train" and args.use_multiprocessing) or (
                    mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(len(data) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(p.imap(preprocess_fn, data, chunksize=chunksize), total=len(data), disable=args.silent, )
                    )
            else:
                self.examples = [preprocess_fn(d) for d in tqdm(data, disable=args.silent)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, losses):

        loss0 = losses[0]
        loss1 = losses[1]

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * loss1 + self.log_vars[1]

        return loss0 + loss1

def load_and_cache_examples(self, data, evaluate=False, no_cache=False, verbose=True, silent=False):
    """
    Creates a T5Dataset from data.

    Utility function for train() and eval() methods. Not intended to be used directly.
    """

    encoder_tokenizer = self.encoder_tokenizer
    decoder_tokenizer = self.decoder_tokenizer
    args = self.args

    if not no_cache:
        no_cache = args.no_cache

    if not no_cache:
        os.makedirs(self.args.cache_dir, exist_ok=True)

    mode = "dev" if evaluate else "train"

    if self.args.use_hf_datasets:
        dataset = load_hf_dataset(data, encoder_tokenizer, decoder_tokenizer, self.args)
        return dataset
    else:
        if args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(encoder_tokenizer, decoder_tokenizer, args, data, mode)
        else:
            if args.model_type in ["bart", "mbart", "marian"]:
                return SimpleSummarizationDataset(encoder_tokenizer, self.args, data, mode)
            else:
                return Seq2SeqDataset(encoder_tokenizer, decoder_tokenizer, self.args, data, mode, )


def train_loop(
        self, train_dataset, output_dir, show_running_loss=True, eval_data=None, verbose=True, **kwargs,
):
    """
    Trains the model on train_dataset.

    Utility function to be used by the train_model() method. Not intended to be used directly.
    """

    if use_crf:  # global value
        self.ner_hidden2tag = nn.Linear(len(self.encoder_tokenizer.vocab),
                                        len(idx2tag)).to(self.device)  # num_ner_labels is the type sum of ner labels: TO or BIO etc
        self.num_ner_labels = len(idx2tag)
        # CRF
        self.CRF_model = CRF(len(idx2tag), batch_first=True).to(self.device)
        self.loss_wrapper = MultiTaskLossWrapper(task_num=2)

    model = self.model
    args = self.args

    tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=self.args.dataloader_num_workers,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = []
    custom_parameter_names = set()
    for group in self.args.custom_parameter_groups:
        params = group.pop("params")
        custom_parameter_names.update(params)
        param_group = {**group}
        param_group["params"] = [p for n, p in model.named_parameters() if n in params]
        optimizer_grouped_parameters.append(param_group)

    for group in self.args.custom_layer_parameters:
        layer_number = group.pop("layer")
        layer = f"layer.{layer_number}."
        group_d = {**group}
        group_nd = {**group}
        group_nd["weight_decay"] = 0.0
        params_d = []
        params_nd = []
        for n, p in model.named_parameters():
            if n not in custom_parameter_names and layer in n:
                if any(nd in n for nd in no_decay):
                    params_nd.append(p)
                else:
                    params_d.append(p)
                custom_parameter_names.add(n)
        group_d["params"] = params_d
        group_nd["params"] = params_nd

        optimizer_grouped_parameters.append(group_d)
        optimizer_grouped_parameters.append(group_nd)

    if not self.args.train_custom_parameters_only:
        optimizer_grouped_parameters.extend(
            [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        )

    warmup_steps = math.ceil(t_total * args.warmup_ratio)
    args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

    # TODO: Use custom optimizer like with BertSum?
    if args.optimizer == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer == "Adafactor":
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adafactor_eps,
            clip_threshold=args.adafactor_clip_threshold,
            decay_rate=args.adafactor_decay_rate,
            beta1=args.adafactor_beta1,
            weight_decay=args.weight_decay,
            scale_parameter=args.adafactor_scale_parameter,
            relative_step=args.adafactor_relative_step,
            warmup_init=args.adafactor_warmup_init,
        )
        print("Using Adafactor for T5")
    else:
        raise ValueError(
            "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                args.optimizer
            )
        )

    if args.scheduler == "constant_schedule":
        scheduler = get_constant_schedule(optimizer)

    elif args.scheduler == "constant_schedule_with_warmup":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    elif args.scheduler == "linear_schedule_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    elif args.scheduler == "cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total,
            num_cycles=args.cosine_schedule_num_cycles,
        )

    elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total,
            num_cycles=args.cosine_schedule_num_cycles,
        )

    elif args.scheduler == "polynomial_decay_schedule_with_warmup":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total,
            lr_end=args.polynomial_decay_schedule_lr_end,
            power=args.polynomial_decay_schedule_power,
        )

    else:
        raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

    if (
            args.model_name
            and os.path.isfile(os.path.join(args.model_name, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name, "scheduler.pt")))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info(" Training started")

    global_step = 0
    training_progress_scores = None
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0)
    epoch_number = 0
    best_eval_metric = None
    early_stopping_counter = 0
    steps_trained_in_current_epoch = 0
    epochs_trained = 0

    if args.model_name and os.path.exists(args.model_name):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name.split("/")[-1].split("-")
            if len(checkpoint_suffix) > 2:
                checkpoint_suffix = checkpoint_suffix[1]
            else:
                checkpoint_suffix = checkpoint_suffix[-1]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
            )

            logger.info("   Continuing training from checkpoint, will skip to saved global_step")
            logger.info("   Continuing training from epoch %d", epochs_trained)
            logger.info("   Continuing training from global step %d", global_step)
            logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("   Starting fine-tuning.")

    if args.evaluate_during_training:
        training_progress_scores = self._create_training_progress_scores(**kwargs)

    if args.wandb_project:
        wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
        wandb.watch(self.model)

    if args.fp16:
        from torch.cuda import amp

        scaler = amp.GradScaler()

    for current_epoch in train_iterator:
        model.train()
        if epochs_trained > 0:
            epochs_trained -= 1
            continue
        train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
            disable=args.silent,
            mininterval=0,
        )
        for step, batch in enumerate(batch_iterator):
            # if step == 0:
            #     print("\n****** inside batch loop *****\n")
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            # batch = tuple(t.to(device) for t in batch)

            inputs = self._get_inputs_dict(batch)
            if args.fp16:
                # if step == 0:
                    # print("\n****** inside FP16 *****\n")
                with amp.autocast():
                    outputs = model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]
            else:
                # if step == 0:
                    # print("\n****** inside normal execution *******\n")
                outputs = model(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]
            ner_tags, ner_mask = batch["target_ner_tags"], batch["target_ner_mask"]
            ner_logits = self.ner_hidden2tag(outputs[1].to(self.device)) # (batch, seq, vocab_size) --> (batch, seq, num_ner_labels)
            ner_loss_list = self.CRF_model(ner_logits.to(self.device), ner_tags.to(self.device),
                                           ner_mask.type(torch.ByteTensor).to(self.device),
                                           reduction='none')
            ner_loss = torch.mean(-ner_loss_list)
            # ner_predict = self.CRF_model.decode(ner_logits, ner_mask.type(torch.ByteTensor))

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                ner_loss = ner_loss.mean()

            current_loss = loss.item() + ner_loss.item()

            if show_running_loss:
                batch_iterator.set_description(
                    f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                combined_loss = self.loss_wrapper([loss, ner_loss])
                scaler.scale(combined_loss).backward()
                # scaler.scale(loss).backward(retain_graph=True)
                # loss.backward(retain_graph=True)
                # scaler.scale(ner_loss).backward()
            else:
                combined_loss = self.loss_wrapper([loss, ner_loss])
                combined_loss.backward()
                # loss.backward()
                # loss.backward(retain_graph=True)
                # ner_loss.backward()
            # tr_loss += loss.item()
            tr_loss += current_loss
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                if args.optimizer == "AdamW":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    if args.wandb_project or self.is_sweeping:
                        wandb.log(
                            {
                                "Training loss": current_loss,
                                "lr": scheduler.get_last_lr()[0],
                                "global_step": global_step,
                            }
                        )

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                    self.save_model(output_dir_current, optimizer, scheduler, model=model)

                if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                ):
                    # Only evaluate when single GPU otherwise metrics may not average well
                    results = self.eval_model(
                        eval_data,
                        verbose=verbose and args.evaluate_during_training_verbose,
                        silent=args.evaluate_during_training_silent,
                        **kwargs,
                    )
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                    output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                    if args.save_eval_checkpoints:
                        self.save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

                    training_progress_scores["global_step"].append(global_step)
                    training_progress_scores["train_loss"].append(current_loss)
                    for key in results:
                        training_progress_scores[key].append(results[key])
                    report = pd.DataFrame(training_progress_scores)
                    report.to_csv(
                        os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                    )

                    if args.wandb_project or self.is_sweeping:
                        wandb.log(self._get_last_metrics(training_progress_scores))

                    if not best_eval_metric:
                        best_eval_metric = results[args.early_stopping_metric]
                        if args.save_best_model:
                            self.save_model(
                                args.best_model_dir, optimizer, scheduler, model=model, results=results
                            )
                    if best_eval_metric and args.early_stopping_metric_minimize:
                        if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                            best_eval_metric = results[args.early_stopping_metric]
                            if args.save_best_model:
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                            early_stopping_counter = 0
                        else:
                            if args.use_early_stopping:
                                if early_stopping_counter < args.early_stopping_patience:
                                    early_stopping_counter += 1
                                    if verbose:
                                        logger.info(f" No improvement in {args.early_stopping_metric}")
                                        logger.info(f" Current step: {early_stopping_counter}")
                                        logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                else:
                                    if verbose:
                                        logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                        logger.info(" Training terminated.")
                                        train_iterator.close()
                                    return (
                                        global_step,
                                        tr_loss / global_step
                                        if not self.args.evaluate_during_training
                                        else training_progress_scores,
                                    )
                    else:
                        if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                            best_eval_metric = results[args.early_stopping_metric]
                            if args.save_best_model:
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                            early_stopping_counter = 0
                        else:
                            if args.use_early_stopping:
                                if early_stopping_counter < args.early_stopping_patience:
                                    early_stopping_counter += 1
                                    if verbose:
                                        logger.info(f" No improvement in {args.early_stopping_metric}")
                                        logger.info(f" Current step: {early_stopping_counter}")
                                        logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                else:
                                    if verbose:
                                        logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                        logger.info(" Training terminated.")
                                        train_iterator.close()
                                    return (
                                        global_step,
                                        tr_loss / global_step
                                        if not self.args.evaluate_during_training
                                        else training_progress_scores,
                                    )
                    model.train()

        epoch_number += 1
        output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

        if args.save_model_every_epoch or args.evaluate_during_training:
            os.makedirs(output_dir_current, exist_ok=True)

        if args.save_model_every_epoch:
            self.save_model(output_dir_current, optimizer, scheduler, model=model)

        if args.evaluate_during_training and args.evaluate_each_epoch:
            results = self.eval_model(
                eval_data,
                verbose=verbose and args.evaluate_during_training_verbose,
                silent=args.evaluate_during_training_silent,
                **kwargs,
            )

            if args.save_eval_checkpoints:
                self.save_model(output_dir_current, optimizer, scheduler, results=results)

            training_progress_scores["global_step"].append(global_step)
            training_progress_scores["train_loss"].append(current_loss)
            for key in results:
                training_progress_scores[key].append(results[key])
            report = pd.DataFrame(training_progress_scores)
            report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)

            if args.wandb_project or self.is_sweeping:
                wandb.log(self._get_last_metrics(training_progress_scores))

            if not best_eval_metric:
                best_eval_metric = results[args.early_stopping_metric]
                if args.save_best_model:
                    self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
            if best_eval_metric and args.early_stopping_metric_minimize:
                if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                    best_eval_metric = results[args.early_stopping_metric]
                    if args.save_best_model:
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                    early_stopping_counter = 0
                else:
                    if args.use_early_stopping and args.early_stopping_consider_epochs:
                        if early_stopping_counter < args.early_stopping_patience:
                            early_stopping_counter += 1
                            if verbose:
                                logger.info(f" No improvement in {args.early_stopping_metric}")
                                logger.info(f" Current step: {early_stopping_counter}")
                                logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                        else:
                            if verbose:
                                logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                logger.info(" Training terminated.")
                                train_iterator.close()
                            return (
                                global_step,
                                tr_loss / global_step
                                if not self.args.evaluate_during_training
                                else training_progress_scores,
                            )
            else:
                if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                    best_eval_metric = results[args.early_stopping_metric]
                    if args.save_best_model:
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                    early_stopping_counter = 0
                else:
                    if args.use_early_stopping and args.early_stopping_consider_epochs:
                        if early_stopping_counter < args.early_stopping_patience:
                            early_stopping_counter += 1
                            if verbose:
                                logger.info(f" No improvement in {args.early_stopping_metric}")
                                logger.info(f" Current step: {early_stopping_counter}")
                                logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                        else:
                            if verbose:
                                logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                logger.info(" Training terminated.")
                                train_iterator.close()
                            return (
                                global_step,
                                tr_loss / global_step
                                if not self.args.evaluate_during_training
                                else training_progress_scores,
                            )

    return (
        global_step,
        tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores,
    )


dataset = sys.argv[1]
task = sys.argv[2]
phr_sen = "_phrase" if sys.argv[3] == 'phrase' else ""
run = sys.argv[4]
model_size = sys.argv[5]

use_crf = True
# use_crf = False
if use_crf:
    from simpletransformers.seq2seq.seq2seq_model import Seq2SeqModel

    Seq2SeqModel.load_and_cache_examples = load_and_cache_examples
    Seq2SeqModel.train = train_loop

    idx2tag, tag2idx = [], {}
    if task == "TASD":
        idx2tag = ["[START]", "[PAD]", "[END]", "B-tgt", "I-tgt", "B-asp", "I-asp", "B-pol", "B-sep", "I-sep",
                   "B-opsep", "I-opsep"]
    elif task == "ASD":
        idx2tag = ["[START]", "[PAD]", "[END]", "B-asp", "I-asp", "B-pol", "B-sep", "I-sep", "B-opsep", "I-opsep"]
    elif task == "AD":
        idx2tag = ["[START]", "[PAD]", "[END]", "B-asp", "I-asp", "B-opsep", "I-opsep"]
    elif task == "TSD":
        idx2tag = ["[START]", "[PAD]", "[END]", "B-tgt", "I-tgt", "B-pol", "B-sep", "I-sep", "B-opsep", "I-opsep"]
    elif task == "TD":
        idx2tag = ["[START]", "[PAD]", "[END]", "B-tgt", "I-tgt", "B-opsep", "I-opsep"]
    elif task == "TAD":
        idx2tag = ["[START]", "[PAD]", "[END]", "B-tgt", "I-tgt", "B-asp", "I-asp", "B-sep", "I-sep", "B-opsep",
                   "I-opsep"]

    if idx2tag is not None and len(idx2tag) > 0:
        tag2idx = {v: i for i, v in enumerate(idx2tag)}
    else:
        print("idx2tag not defined properly for the task")
        exit(0)

print(f"dataset: {dataset}\ntask: {task}\nphr_sen: {phr_sen}\nrun: {run}\nmodel_size:{model_size}")

train = False
# train = True
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

print(model_args.output_dir)
# Initialize model

if train:
    train_df = pd.read_csv(f'data/{dataset}/train_{task}{phr_sen}.csv', na_filter=False)
    train_df = train_df.sample(frac=1)
    train_df = train_df[["input_text", "target_text"]]

    target_texts = train_df["target_text"].values.tolist()
    ner_tags = construct_ner_tags(target_texts=target_texts, ner_tags=[])
    train_df["ner_tags"] = ner_tags

    print(train_df.head())

    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name=f"facebook/bart-{model_size}",
        args=model_args,
        use_cuda=False if not torch.cuda.is_available() else True
    )

    model.train_model(train_df)

    print(model.predict(["The lemon chicken tasted like sticky sweet donuts and the honey walnut prawns , the few they "
                         "actually give you . . . . . were not good ."]))

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

    with open(f'results/{task}{run}_{dir_prefix}/truth.pkl', "wb") as f:
        pickle.dump(truth, f)
    with open(f'results/{task}{run}_{dir_prefix}/preds.pkl', "wb") as f:
        pickle.dump(preds, f)

else:
    with open(f'results/{task}{run}_{dir_prefix}/truth.pkl', "rb") as f:
        truth = pickle.load(f)
    with open(f'results/{task}{run}_{dir_prefix}/preds.pkl', "rb") as f:
        preds = pickle.load(f)

# exit(0)

if dataset in ['semeval-2015', 'semeval-2016']:
    if task == "TASD":
        convert_pred_to_TAS_format(truth, preds)
    elif task == "ASD" or task == "AD":
        ASD(truth, preds, f"evaluation_for_AD_TD_TAD/ABSA{15 if '15' in dataset else 16}_Restaurants_Test.xml")
    elif task == "TSD" or task == "TD":
        TSD(truth, preds, f"evaluation_for_AD_TD_TAD/ABSA{15 if '15' in dataset else 16}_Restaurants_Test.xml")
    else:
        TAD(truth, preds, f"evaluation_for_AD_TD_TAD/ABSA{15 if '15' in dataset else 16}_Restaurants_Test.xml")