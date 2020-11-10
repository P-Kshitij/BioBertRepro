import transformers

DATASET_LIST = [
    'BC2GM',
    'BC4CHEMD',
    'BC5CDR-chem',
    'BC5CDR-disease',
    'JNLPBA',
    'linnaeus',
    'NCBI-disease',
    's800'
]
DATASET_LIST_DISEASE = [
    'BC5CDR-disease',
    'NCBI-disease'
]
DATASET_LIST_CHEM = [
    'BC4CHEMD',
    'BC5CDR-chem'
]
DATASET_LIST_GENE = [
    'BC2GM',
    'JNLPBA'
]
DATASET_LIST_SPECIES = [
    'linnaeus',
    's800'
]

TRAINING_FILE = "../input/NERdataset_preproc/NCBI-disease/devel.tsv"
TEST_FILE = "../input/NERdataset_preproc/NCBI-disease/test.tsv"
METADATA_PATH = "meta.bin"
BERT_PATH = "../input/bert-base-uncased/"
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
MODEL_PATH = "../models/model.bin"
MAX_LEN = 512
EPOCHS = 10
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)