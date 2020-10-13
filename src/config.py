import transformers

TRAINING_FILE = "../input/NERdataset_preproc/NCBI-disease/devel.tsv"
BERT_PATH = "../input/bert-base-uncased/"
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
MODEL_PATH = "../models/model.bin"
MAX_LEN = 512
EPOCHS = 10
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)