import sys

import torch.cuda
from simpletransformers.t5 import T5Model
from sklearn.model_selection import train_test_split

import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# dataset = 'semeval-2014'
# task = 'ASD'
model_size = "base"
# model_size = "large"

dataset = sys.argv[1]
task = sys.argv[2]
phr_sen = "_phrase" if sys.argv[3] == 'phrase' else ""
run = sys.argv[4]
print(f"dataset: {dataset}\ntask: {task}\nphr_sen: {phr_sen}\nrun: {run}")

# dir_prefix = f"{dataset}_{model_size}"
dir_prefix = f"{dataset}{phr_sen}"
train_df = pd.read_csv(f'data/{dataset}/train_{task}{phr_sen}.csv')

train_df = train_df.sample(frac=1)
eval_df = None
# eval_df = pd.read_csv(f"data/{dataset}/valid_{task}.csv")
train_df, eval_df = train_test_split(train_df, test_size=0.1)

print("\n".join(train_df["target_text"].head().tolist()))

model_args = {
    "max_seq_length": 128,
    "train_batch_size": 16,
    "eval_batch_size": 64,
    "num_train_epochs": 50,
    "evaluate_during_training": True if eval_df is not None else False,
    "evaluate_during_training_steps": (len(train_df) / 16) if eval_df is not None else None,
    "evaluate_during_training_verbose": True if eval_df is not None else False,
    "manual_seed": 1234,
    "learning_rate": 4e-5,

    "use_multiprocessing": False,
    "use_multiprocessing_for_evaluation": False,
    "fp16": False,

    "save_steps": -1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,

    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "output_dir": f"results/{task}{run}_{dir_prefix}/",
    "best_model_dir": f"results/{task}{run}_{dir_prefix}/best_model/" if eval_df is not None else None,

    "wandb_project": f"{task}_{dir_prefix}",
}
print(f"\n\n************* output dir: {model_args['output_dir']} ******************\n\n")
model = T5Model("t5", f"t5-{model_size}", args=model_args, use_cuda=False if not torch.cuda.is_available() else True)

model.train_model(train_df, eval_data=eval_df)
