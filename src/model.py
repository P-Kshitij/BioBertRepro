import transformers
import config
import torch
import torch.nn as nn

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1,1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class NERModel(nn.Module):
    def __init__(self, num_tags):
        super(NERModel, self).__init__()
        self.num_tags = num_tags
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tags)

    def forward(ids, mask, token_type_ids, target_tags):
        o1, _  =  self.bert(
            ids,
            mask = mask,
            token_type_ids = token_type_ids
        )
        bo_tags = self.drop(o1)

        tags = self.out_tag(bo_tags)
        loss = loss_fn(tags, target_tags, mask, self.num_tags)
    
        return tags, loss
