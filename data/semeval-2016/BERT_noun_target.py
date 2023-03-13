import sys

from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)
import pandas as pd
import logging
import torch
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = pd.read_csv("TASD_train_noun_target.csv")
train_df.columns = ["text_a", "text_b", "labels"]

eval_df = pd.read_csv("TASD_test_noun_target.csv")
eval_df.columns = ["text_a", "text_b", "labels"]

model_args = ClassificationArgs(num_train_epochs=10)

# task = sys.argv[1]
task = 'TASD'
run = sys.argv[1]
# run = 1
dir_prefix = "noun_"
is_train = False if int(sys.argv[2]) == 0 else True
best_model = True
lr = float(sys.argv[3])

model_args = {
    "max_seq_length": 128,
    "train_batch_size": 16,
    "eval_batch_size": 64,
    "num_train_epochs": 20,
    "evaluate_during_training": True if eval_df is not None else False,
    "evaluate_during_training_steps": (len(train_df) / 16) if eval_df is not None else None,
    "evaluate_during_training_verbose": True if eval_df is not None else False,
    "manual_seed": 1234,
    "learning_rate": lr,
    # "scheduler": "polynomial_decay_schedule_with_warmup",
    "warmup_ratio": 0.50,

    "use_multiprocessing": False,
    "use_multiprocessing_for_evaluation": False,
    "fp16": False,

    # "use_early_stopping": True,

    "save_steps": -1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,

    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "output_dir": f"noun_results/{task}{run}_{dir_prefix}/",
    "best_model_dir": f"noun_results/{task}{run}_{dir_prefix}/best_model/" if eval_df is not None else None,

    "wandb_project": f"{task}{run}_{dir_prefix}",
}

# Create a ClassificationModel
model = ClassificationModel("bert", "bert-base-uncased", args=model_args,
                            use_cuda=False if not torch.cuda.is_available() else True
                            # ,cuda_device=6
                            # , num_labels=2,
                            # weight=[1, 3]
                            )

if is_train:
    # Train the model
    model.train_model(train_df=train_df, eval_df=eval_df)
else:
    model = ClassificationModel("bert", f"noun_results/{task}{run}_{dir_prefix}/{'best_model/' if best_model else ''}",
                                args=model_args,
                                use_cuda=False if not torch.cuda.is_available() else True)
# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(
    eval_df
)

# Prepare the data for testing
to_predict = [
    [text_a, text_b] for text_a, text_b in zip(eval_df["text_a"].tolist(), eval_df["text_b"].tolist())
]
truth = eval_df["labels"].tolist()

# Get the model predictions
preds = model.predict(to_predict)

# print(preds[0])

df = pd.DataFrame(eval_df["text_a"].tolist(), columns=["text_a"])
df["text_b"] = eval_df["text_b"].tolist()
df["Truth"] = truth
df["Pred"] = preds[0]
df.to_csv(f"noun_results/{task}{run}_{dir_prefix}/{'best_model/' if best_model else ''}" + 'TASD1_output.csv', header=True, index=False)
print(f"noun_results/{task}{run}_{dir_prefix}/{'best_model/' if best_model else ''}" + 'TASD1_output.csv')

print(classification_report(truth, preds[0]))

# with open(f"noun_results/{task}{run}_{dir_prefix}/{'best_model/' if best_model else ''}" + 'TASD1_truth.pkl', "wb") as f:
#     pickle.dump(truth, f)
# with open(f"noun_results/{task}{run}_{dir_prefix}/{'best_model/' if best_model else ''}" + 'TASD1_preds.pkl', "wb") as f:
#     pickle.dump(preds, f)