import pandas as pd
from ast import literal_eval
import joblib
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

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

def evaluate(test_dataloader, model, device, num_tags, grouped_entities=False):
    model.eval()
    tags_ytrue = []
    tags_ypred = []
    with torch.no_grad():
        for data in tqdm(test_dataloader, total=len(test_dataloader)):
            for k,v in data.items():
                data[k] = v.to(device)
            data_without_group_masks = {k:v for k,v in d.items() if k!='group_masks'}
            logits, _ = model(**data)
            if grouped_entities==True:
                for i,text in enumerate(data["ids"].cpu().numpy()):
                    grouped_text = config.TOKENIZER.decode(
                        text,
                        skip_special_tokens = True
                    )
                    ungrouped_text = config.TOKENIZER.convert_ids_to_tokens(
                        text,
                        skip_special_tokens = True
                    )
                    print(grouped_text)
                    print(ungrouped_text)
                    print(len(ungrouped_text))
                    print(len(data["mask"].cpu().numpy()[i]))
                    print('target_tags: ', data['target_tags'][0,data['mask'].cpu().numpy()])
                    break
                    #print((np.array(grouped_text))[data["mask"].cpu().numpy()[i]])
                    #print(np.array(grouped_text)))

            tags_pred = logits.argmax(2).cpu().numpy()
            mask_np = data['mask'].cpu().numpy()
            target_tags_np = data['target_tags'].cpu().numpy()
            for idx, arr in enumerate(tags_pred):
                if idx==0:
                    print(list(arr[mask_np[idx,:]==1])[1:-1])
                    print(len(list(arr[mask_np[idx,:]==1])[1:-1]))
                tags_ypred.extend(list(arr[mask_np[idx,:]==1])[1:-1])
                tags_ytrue.extend(list(target_tags_np[idx, mask_np[idx,:]==1])[1:-1])
            assert(len(tags_ytrue)==len(tags_ypred))
            return tags_ypred, tags_ytrue

    return tags_ypred, tags_ytrue
    




if __name__ == "__main__":
    meta_data = joblib.load(config.METADATA_PATH)
    for v in meta_data.values():
        enc_tags = v
    num_tags = len(list(enc_tags.classes_))
    sentences, tags = preprocess_data(config.TEST_FILE, enc_tags)
    test_dataloader = get_dataloader(sentences, tags)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NERModel(num_tags)
    model.load_state_dict(torch.load(config.MODEL_PATH,map_location=device))
    tags_ypred, tags_ytrue = evaluate(test_dataloader, model, device, num_tags, grouped_entities=True)
    tags_ypred = enc_tags.inverse_transform(tags_ypred)
    tags_ytrue = enc_tags.inverse_transform(tags_ytrue)
    print(tags_ytrue,tags_ypred)
    print(classification_report(tags_ytrue, tags_ypred))