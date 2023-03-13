#!/bin/bash 

CUDA_VISIBLE_DEVICES=4 python T5_SemEval_Train.py semeval-2016 TASD sentence 3
CUDA_VISIBLE_DEVICES=4 python T5_SemEval_Test.py semeval-2016 TASD sentence 3 False
CUDA_VISIBLE_DEVICES=4 python merge_pred_files.py semeval-2016 TASD sentence 3 False
./evaluate.sh 3_semeval-2016

CUDA_VISIBLE_DEVICES=4 python T5_SemEval_Train.py semeval-2016 TASD sentence 4
CUDA_VISIBLE_DEVICES=4 python T5_SemEval_Test.py semeval-2016 TASD sentence 4 False
CUDA_VISIBLE_DEVICES=4 python merge_pred_files.py semeval-2016 TASD sentence 4 False
./evaluate.sh 4_semeval-2016
