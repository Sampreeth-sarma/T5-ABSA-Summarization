import pandas as pd
# task = 'AD'
task = 'ASD'

df = pd.read_csv('train_TAS.tsv', sep='\t')
df = df[df.yes_no == 1]
df['input_text'] = df.sentence
if task == 'AD':
    df['target_text'] = df.groupby(['input_text'])['aspect_sentiment'].transform(lambda x: ','.join(set([' '.join(each.split()[:2]).strip() for each in x])))
else:
    df['target_text'] = df.groupby(['input_text'])['aspect_sentiment'].transform(lambda x: ','.join(set([each.strip() for each in x])))

new_df = df[['input_text', 'target_text']]
new_df["prefix"] = f"Semeval {task}"

new_df.drop_duplicates(inplace=True)
print(new_df)
new_df.to_csv(f'train_{task}.csv', header=True, index=False)

df = pd.read_csv('test_TAS.tsv', sep='\t')
df = df[df.yes_no == 1]
df['input_text'] = df.sentence

if task == 'AD':
    df['target_text'] = df.groupby(['input_text'])['aspect_sentiment'].transform(lambda x: ','.join(set([' '.join(each.split()[:2]).strip() for each in x])))
else:
    df['target_text'] = df.groupby(['input_text'])['aspect_sentiment'].transform(lambda x: ','.join(set([each.strip() for each in x])))

new_df = df[['input_text', 'target_text']]
new_df["prefix"] = f"Semeval {task}"

new_df.drop_duplicates(inplace=True)
print(new_df)
new_df.to_csv(f'test_{task}.csv', header=True, index=False)
