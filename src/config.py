import transformers

TRAINING_FILE = "../input/NERdataset_preproc/NCBI-disease/"
BERT_PATH = "../input/bert-base-uncased/"
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
MODEL_PATH = "../models/"
MAX_LEN = 512
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)