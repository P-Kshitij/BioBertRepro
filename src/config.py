import transformers

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

# Make changes here to choose another dataset list
DATASET_LIST = DATASET_LIST_DISEASE

DATASET_PATH = "../input/NERdataset_preproc/"
TRAINING_FILE = "train_dev.tsv"
TEST_FILE = "test.tsv"
METADATA_PATH = "meta.bin"
BERT_PATH = "../input/bert-base-uncased/"
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
MODEL_PATH = "../models/model.bin"
DRIVE_MODEL_PATH = "/content/gdrive/My Drive/BioBert/models/10-11-20_full_disease/"


MAX_LEN = 512
EPOCHS = 10
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)