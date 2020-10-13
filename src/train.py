import pandas as pd
from sklearn import preprocessing
from ast import literal_eval

import config
import dataset
import engine
from model import NERModel

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df['labels'] = df['labels'].apply(literal_eval)
    enc_tags = preprocessing.LabelEncoder()
    
    all_labels = []
    for e in df['labels'][:5]:
        all_labels.extend(e)

    enc_tags.fit(all_labels)
    df['labels'] = df['labels'].apply(enc_tags.transform)
    print(df[:5])

if __name__ == "__main__":
    preprocess_data(config.TRAINING_FILE)