import gc
import torch
from transformers.file_utils import ModelOutput
from transformers import BertModel
from torch import nn


class BERTRNN(nn.Module):

    def __init__(self, model_name='bert-base-uncased', device='cuda', num_classes=3):
        super(BERTRNN, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # RNN
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True).to(device)
        # FC
        self.fc = nn.Linear(256*2, num_classes).to(device)
        self.softmax = nn.LogSoftmax(dim=1).to(device)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        lstm_output, (h, c) = self.lstm(outputs[0])
        hidden = torch.cat((lstm_output[:, -1, :256],lstm_output[:, 0, 256:]), dim=-1)
        x = self.fc(hidden.view(-1,256*2))
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
