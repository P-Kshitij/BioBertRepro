# BioBERT Repro
![Python](https://img.shields.io/badge/python-v3.6.9-blue.svg) ![Pytorch](https://img.shields.io/badge/PyTorch-v1.7.0-blueviolet) ![cuda](https://img.shields.io/badge/CUDA-v10.1-green) ![transformers](https://img.shields.io/badge/transformers-v3.5.1-blue)
---
Trying to reproduce BioBERT([paper link](https://arxiv.org/abs/1901.08746)) results for the NER task. The [original work](https://github.com/dmis-lab/biobert) of the paper is in Tensorflow. This repo is made using 🤗HuggingFace transformers library and Pytorch.
### Credits:
* Code and project workflow heavily inspired by [Abhishek Thakur](https://www.youtube.com/user/abhisheksvnit)'s tutorials.
* 🤗[HuggingFace](https://huggingface.co/)'s Transformers documentation
* Evaluation details and other specifics taken from original [paper](https://arxiv.org/abs/1901.08746)
### Steps: 
---
Run `setup.sh` to download the NER datasets of the paper and do preprocessing.
```bash
bash setup.sh
```
### Finetuning on NER-datasets:
___
Run `train.py` to finetune the model on disease-datasets of the paper. Additionally to finetune on other datasets, make changes to `config.py`. There is an optional argument of secondary folderpath which can be when working on a remote server. When invoked, apart from saving the metadata and model in the local repository, it is also saved at a secondary path (For eg a G-drive path when working on a Google Colab server). 
```bash
cd src
python train.py [-f(optional) "SECONDARY_FOLDERPATH"]
```
The best model (according to val score) will be saved in `models/` and metadata (containing label encoder object) will be saved in `src/`. The same will be done for the SECONDARY_FOLDERPATH if provided.

### Evaluating a finetuned model:
___
Make sure your trained model is in `models/` and your metadata dict containing the `LabelEncoder` object is in `src/`. Run `evaluate.py`. There is optional argument `-g` which if invoked, will evaluate all metrics on exact entity-level matching; use `python evaluate.py --help` for more info on the arguments. You should see a classification report generated with the help of [`seqeval`](https://github.com/chakki-works/seqeval).
```bash
cd src
python evaluate.py [-g]
```
### Results:
___
Classification report of the NER task on a model finetuned for 10 epochs on the disease-datasets:

[![CR-10epoch-full-disease.png](https://i.postimg.cc/Fz4jL32Q/CR-10epoch-full-disease.png)](https://postimg.cc/tZBZ0Zqr)

<br>Here evaluation is done for single entity-type (disease) and hence the averages are the same.
### Some important points:
---
* In the original paper, they have pretrained on biomedical datasets and then finetuned for downstream tasks. Here we only finetune a `bert-base-uncased` model without any pretraining. The model can be changed in  `config.py` with minimal changes to `model.py`.
* `evaluate.py` contains custom methods by which we carry out entity-level NER exact evaluation (as stated in the paper). Meaning we first convert the predictions on Wordpiece tokens to word-level predictions, and then carry out exact matching to get NER metrics namely F1 score, precision and recall.
