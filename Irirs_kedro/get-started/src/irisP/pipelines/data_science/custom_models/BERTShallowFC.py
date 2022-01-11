import gc
import torch
from transformers.file_utils import ModelOutput
from transformers import BertModel
from torch import nn


class BERTShallowFC(nn.Module):
    def __init__(self, model_name='bert-base-uncased', device='cuda'):
        super(BERTShallowFC, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(768, 3).to(device)  # (BERT CLS size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1).to(device)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        x = self.fc(outputs[0][:, 0, :])
        c = self.softmax(x)

        # Clean cache
        gc.collect()
        torch.cuda.empty_cache()
        del outputs

        # Compute loss
        loss = None
        if labels is not None:
            ce_loss = nn.CrossEntropyLoss()
            loss = ce_loss(c, labels)

        return ModelOutput({
            'loss': loss,
            'last_hidden_state': c
        })
