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
                    print(data['group_masks'][0])
                    print(type(data['group_masks']))
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
            group_masks = data['group_masks'].cpu().numpy()
            for idx, arr in enumerate(tags_pred):
                # if idx==0:
                #     print(list(arr[mask_np[idx,:]==1])[1:-1])
                #     print(len(list(arr[mask_np[idx,:]==1])[1:-1]))
                pred_tags = arr[mask_np[idx,:]==1][1:-1]
                target_tags = (target_tags_np[idx, mask_np[idx,:]==1])[1:-1]
                assert(len(pred_tags)==len(target_tags))
                if grouped_entities == True:
                    group_mask_idx = group_masks[idx, group_masks[idx]!=0]
                    grouped_tags_ypred = []
                    grouped_tags_ytrue = []
                    assert(len(group_mask_idx)==len(pred_tags))
                    i = 0
                    while(i < len(group_mask_idx)
                    ):
                        group_idx = group_mask_idx[i]
                        target_group_tags = set()
                        pred_group_tags = set()
                        if idx==0:
                            print('starting i',i)
                        while(i<len(group_mask_idx) and group_mask_idx[i]==group_idx):
                            if idx==0:
                                print(i, group_mask_idx[i], group_idx)
                            pred_group_tags.add(pred_tags[i])
                            target_group_tags.add(target_tags[i])
                            i+=1
                        if idx==0:
                            print('ending i',i)
                        # Sanity check to ensure that when we rebuild from tokens, we get the same tags.
                        # Note that we just did reverse of this transform in dataset.py lines (29)
                        assert(len(target_group_tags)==1)
                        grouped_tags_ytrue.append(target_group_tags.pop())
                        if(len(pred_group_tags)==1):
                            # Here we have the same tags for all tokens of a word
                            # Meaninng ['ata' , '##xia'] has tags ['B','B'] meaning we can combine
                            grouped_tags_ypred.append(pred_group_tags.pop())
                        else:
                            # Here we have the different tags for different tokens of a word
                            # Meaninng ['ata' , '##xia'] has tags ['B','O'] meaning we can must treat 
                            # as a wrong answer and add an 'X' 
                            grouped_tags_ypred.append(3)
                    print('idx: ',idx,'len',len(grouped_tags_ypred))
                    print(grouped_tags_ypred)
                    print(grouped_tags_ytrue)
                    print(group_mask_idx)
                    tags_ypred.extend(grouped_tags_ypred)
                    tags_ytrue.extend(grouped_tags_ytrue)
                    assert(len(tags_ytrue)==len(tags_ypred))
                else:   
                    tags_ypred.extend(list(pred_tags))
                    tags_ytrue.extend(list(target_tags))
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
    # tags_ypred = enc_tags.inverse_transform(tags_ypred)
    # tags_ytrue = enc_tags.inverse_transform(tags_ytrue)
    print(tags_ytrue,tags_ypred)
    print(classification_report(tags_ytrue, tags_ypred))