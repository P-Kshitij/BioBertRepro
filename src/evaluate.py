import pandas as pd
from ast import literal_eval
import joblib
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification_report
import numpy as np
import argparse

import config
import dataset
from model import NERModel

def preprocess_data(enc_tags):
    sentences = []
    tags = []
    for dataset_name in config.DATASET_LIST:
        data_path = config.DATASET_PATH + dataset_name + "/" + config.TEST_FILE
        df = pd.read_csv(data_path)
        df['sentence'] = df['sentence'].apply(literal_eval)
        df['labels'] = df['labels'].apply(literal_eval)

        df['labels'] = df['labels'].apply(enc_tags.transform)
        sentences_dataset = list(df['sentence'].values)
        tags_dataset = list(df['labels'].values)
        sentences.extend(sentences_dataset)
        tags.extend(tags_dataset)
    
    sentences = np.array(sentences)
    tags = np.array(tags)
    return sentences, tags

def get_dataloader(sentences, tags):
    test_dataset = dataset.NERdataset(sentences, tags, evalmode = True)
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
            data_without_group_masks = {k:v for k,v in data.items() if k!='group_masks'}
            logits, _ = model(**data_without_group_masks)

            tags_pred = logits.argmax(2).cpu().numpy()
            mask_np = data['mask'].cpu().numpy()
            target_tags_np = data['target_tags'].cpu().numpy()
            group_masks = data['group_masks'].cpu().numpy()
            for idx, arr in enumerate(tags_pred):
                pred_tags = arr[mask_np[idx,:]==1][1:-1]
                target_tags = (target_tags_np[idx, mask_np[idx,:]==1])[1:-1]
                assert(len(pred_tags)==len(target_tags))
                if grouped_entities == True:
                    group_mask_idx = group_masks[idx, group_masks[idx]!=0]
                    grouped_tags_ypred = []
                    grouped_tags_ytrue = []
                    assert(len(group_mask_idx)==len(pred_tags))
                    i = 0
                    while(i < len(group_mask_idx)):
                        group_idx = group_mask_idx[i]
                        target_group_tags = set()
                        pred_group_tags = set()
                        while(i<len(group_mask_idx) and group_mask_idx[i]==group_idx):
                            pred_group_tags.add(pred_tags[i])
                            target_group_tags.add(target_tags[i])
                            i+=1
                        '''
                        Sanity check to ensure that when we rebuild from tokens, we get the same tags.
                        Note that we just did reverse of this transform in dataset.py lines (29)
                        '''
                        assert(len(target_group_tags)==1)
                        grouped_tags_ytrue.append(target_group_tags.pop())
                        if(len(pred_group_tags)==1):
                            '''
                            Here we have the same tags for all tokens of a word
                            Meaninng ['ata' , '##xia'] has tags ['B','B'] meaning we can combine
                            '''
                            grouped_tags_ypred.append(pred_group_tags.pop())
                        else:
                            '''
                            Here we have different tags for different tokens of a word
                            Meaning ['ata' , '##xia'] has tags ['B','O'] meaning we can must treat 
                            as a wrong answer and add an '3', which becomes 'X' on decoding 
                            '''
                            grouped_tags_ypred.append(3)
                    # print('idx: ',idx,'len',len(grouped_tags_ypred))
                    # print(grouped_tags_ypred)
                    # print(grouped_tags_ytrue)
                    # print(group_mask_idx)
                    tags_ypred.append(grouped_tags_ypred)
                    tags_ytrue.append(grouped_tags_ytrue)
                    assert(len(grouped_tags_ypred)==len(grouped_tags_ytrue))
                else:   
                    tags_ypred.append(list(pred_tags))
                    tags_ytrue.append(list(target_tags))
                    assert(len(pred_tags)==len(target_tags))
            # Uncomment for quick evaluation on just 8 examples        
            # return tags_ypred, tags_ytrue

    return tags_ypred, tags_ytrue
    
def decode_transform(arr,enc_tags):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 3:
                arr[i][j] = enc_tags.inverse_transform([arr[i][j]])[0]
            elif arr[i][j] == 3:
                arr[i][j] = 'X'
            else:
                raise KeyError(str(arr[i][j])+' as key not found in Label Encoder ')
    return arr

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.version = '1.0'
    my_parser.add_argument('-g','--grouped_entities', action='store_true',help='if used, evaluate all metrics on exact entity-level matching, instead of just wordpiece-level tokens ')
    args = my_parser.parse_args()
    grouped_entities = args.grouped_entities

    meta_data = joblib.load(config.METADATA_PATH)
    enc_tags = meta_data['enc_tags']

    num_tags = len(list(enc_tags.classes_))
    sentences, tags = preprocess_data(enc_tags)
    test_dataloader = get_dataloader(sentences, tags)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NERModel(num_tags)
    model.load_state_dict(torch.load(config.MODEL_PATH,map_location=device))
    tags_ypred, tags_ytrue = evaluate(test_dataloader, model, device, num_tags, grouped_entities=grouped_entities)
    # tags_ypred = enc_tags.inverse_transform(tags_ypred)
    # tags_ytrue = enc_tags.inverse_transform(tags_ytrue)
    tags_ypred = decode_transform(tags_ypred, enc_tags)
    tags_ytrue = decode_transform(tags_ytrue, enc_tags)
    # print(tags_ytrue,tags_ypred)
    print(seqeval_classification_report(tags_ytrue, tags_ypred))