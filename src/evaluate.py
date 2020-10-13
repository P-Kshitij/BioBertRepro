import pandas as pd
from ast import literal_eval
import joblib
import torch
from tqdm import tqdm

import config
import dataset
from model import NERModel

def preprocess_data(data_path, enc_tags):
    df = pd.read_csv(data_path)
    df['sentence'] = df['sentence'].apply(literal_eval)
    df['labels'] = df['labels'].apply(literal_eval)

    df['labels'] = df['labels'].apply(enc_tags.transform)
    sentences = df['sentence'].values
    tags = df['labels'].values

    return sentences, tags

def get_dataloader(sentences, tags):
    test_dataset = dataset.NERdataset(sentences, tags)
    test_dataloader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = config.TEST_BATCH_SIZE
    )
    return test_dataloader

def evaluate(test_dataloader, model, device, num_tags):
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_dataloader, total=len(test_dataloader)):
            for k,v in data.items():
                data[k] = v.to(device)
            logits, _ = model(**data)
        logits = logits.view(-1, num_tags)
        print(logits)
        raise NotImplementedError
    




if __name__ == "__main__":
    meta_data = joblib.load(config.LABEL_ENCODER_PATH)
    for v in meta_data.values():
        enc_tags = v
    print(type(enc_tags))
    num_tags = len(list(enc_tags.classes_))
    sentences, tags = preprocess_data(config.TEST_FILE, enc_tags)
    test_dataloader = get_dataloader(sentences, tags)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NERModel(num_tags)
    model.load_state_dict(torch.load(config.MODEL_PATH,map_location=device))
    evaluate(test_dataloader, model, device, num_tags)