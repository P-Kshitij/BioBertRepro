from torch.utils.data import Dataset
import torch
import config

class NERdataset(Dataset):
    def __init__(self, texts, tags, evalmode=False):
        self.texts = texts
        self.tags = tags
        self.evalmode = evalmode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        tag = self.tags[item]

        ids = []
        target_tags = []
        group_masks = []
        for i,s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens = False
            )
            input_len = len(inputs)
            ids.extend(inputs)
            target_tags.extend([tag[i]]*input_len)
            group_masks.extend([i]*input_len)

        # Truncating to max_len
        ids = ids[:config.MAX_LEN-2]
        target_tags = target_tags[:config.MAX_LEN-2]

        # Adding special tokens 
        ids = [101] + ids + [102]
        target_tags = [0] + target_tags + [0]

        mask = [1]*len(ids)
        token_type_ids = [0]*len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0]*padding_len)
        target_tags = target_tags + ([0]*padding_len)
        mask = mask + ([0]*padding_len)
        token_type_ids = token_type_ids + ([0]*padding_len)

        # Sanity check
        assert(len(ids)==config.MAX_LEN)
        assert(len(target_tags)==config.MAX_LEN)
        assert(len(mask)==config.MAX_LEN)
        assert(len(token_type_ids)==config.MAX_LEN)

        if self.evalmode==True:
            return {
                "ids":torch.tensor(ids, dtype=torch.long),
                "mask":torch.tensor(mask, dtype=torch.long),
                "token_type_ids":torch.tensor(token_type_ids, dtype=torch.long),
                "target_tags": torch.tensor(target_tags, dtype=torch.long),
                "group_masks": torch.tensor(group_masks, dtype=torch.long)
            }
        else: 
            return {
                "ids":torch.tensor(ids, dtype=torch.long),
                "mask":torch.tensor(mask, dtype=torch.long),
                "token_type_ids":torch.tensor(token_type_ids, dtype=torch.long),
                "target_tags": torch.tensor(target_tags, dtype=torch.long)
            }