# T5-ABSA-Summarization
Experimental code and data of paper [Exploring Conditional Text Generation for Aspect-Based Sentiment Analysis](https://aclanthology.org/2021.paclic-1.13.pdf), PACLIC 2021

## Environment
    Python==3.6
    numpy==1.19.5
    torch==1.7.0
    simpletransformers==0.61.4
    transformers==4.18.0

This repository relies on the previous work [TAS-BERT](https://github.com/sysulic/TAS-BERT).

## Repository Description
- All the datasets with the respective folders are present in the [data directory](data)
  - We use a function ([convert_to_text_gen.py](data/semeval-2016/convert_to_text_gen.py)) to convert the data format from and to the TAS format and the Conditional Text Generation or Abstract Summarization format.
- Following the TAS-BERT implementation, we continue to have the [evaluation for AD, TD, TAD](evaluation_for_AD_TD_TAD) directory to facilitate the evaluation for Target Detection, Aspect Detection, and Target-Aspect Joint Detection
- We provide separate .sh files to run  the model on each dataset.
  - [./14_run_ASD.sh](./14_run_ASD.sh), [./sh_run_ASD.sh](./sh_run_ASD.sh)
    - dataset directory name (semeval-2014), task to run (ASD | AD), auxiliary format of the target text (sentence | phrase), and any random number to append to the result directory to distinguish from previous runs (1, 2, ...)
  - [./15_run_ASD.sh](./15_run_ASD.sh), [./16_run_ASD.sh](./16_run_ASD.sh)
    - dataset directory name (semeval-2014), task to run (ASD | AD), auxiliary format of the target text (sentence | phrase), any random number to append to the result directory to distinguish from previous runs (1, 2, ...), and give True to evaluate on the best model based on the dev set from all the epochs or False to evaluate on the last epoch model parameters
- Once you run the model, [results](results), [predictions](predictions), and [runs](runs) directories will be created where all your model results and predictions will be generated

## Running Information
- We have separate scripts to run the T5 and BART models. Based on the model-type you want to evaluate on, please change it in the .sh script files accordingly
- Make sure to give the correct numbers for the available GPUs for CUDA_VISIBLE_DEVICES
- If you use the T5 model, it gives an option to generate as many sentences as you want, and we generated the top 3 most probable sentences for each task. In the results, you can see separate numbers for each of those sentences.
- For SemEval-2014 and SentiHood Datasets, you can train and test for ASD (Aspect Sentiment Joint Detection), and AD (Aspect Detection) tasks
- For SemEval-2015 and SemEval-2016 Datasets, you can train and test for 
  - ASD (Aspect Sentiment Joint Detection), AD (Aspect Detection) tasks
  - TSD (Target Sentiment Joint Detection), TD (Target Detection) tasks
  - TASD (Target Aspect Sentiment Joint Detection), TAD (Target Aspect Detection) tasks

_**Note**_: _You can change other parameters in the train and test script files if needed_


## Citation
If you use this code or method in your work, please cite our paper: 

        @inproceedings{chebolu-etal-2021-exploring,
        title = "Exploring Conditional Text Generation for Aspect-Based Sentiment Analysis",
        author = "Chebolu, Siva Uday Sampreeth  and
          Dernoncourt, Franck  and
          Lipka, Nedim  and
          Solorio, Thamar",
        booktitle = "Proceedings of the 35th Pacific Asia Conference on Language, Information and Computation",
        month = "11",
        year = "2021",
        address = "Shanghai, China",
        publisher = "Association for Computational Lingustics",
        url = "https://aclanthology.org/2021.paclic-1.13",
        pages = "119--129",
    }

  

    