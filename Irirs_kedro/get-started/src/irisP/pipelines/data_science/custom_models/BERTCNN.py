import gc
import torch
from transformers.file_utils import ModelOutput
from transformers import BertModel
from torch import nn


class BERTCNN(nn.Module):

    def __init__(self,
                 model_name='bert-base-uncased',
                 device='cuda',
                 max_length=36,
                 num_classes=3):

        super(BERTCNN, self).__init__()
        self.bert = BertModel.from_pretrained(model_name).to(device)

        # 768: BERT CLS size
        # padding='valid': same as padding=0

        self.conv = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 768), padding='valid').to(device)
        self.relu = nn.ReLU()

        # change the kernel size either to (3,1), e.g. 1D max pooling
        # or remove it altogether
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=1).to(device)
        self.dropout = nn.Dropout(0.1)

        # be careful here, this needs to be changed according to your max pooling
        # without pooling: 443, with 3x1 pooling: 416
        # Size after conv = BERT max length - 3 + 1
        # Size after pool = Size after conv - 3 + 1
        # (BERT max length - 3 + 1) - 3 + 1 == BERT max length - 4
        # (kernel_size * (BERT max length - 4), num. classes)
        # FC
        self.fc = nn.Linear(13 * (max_length - 4), num_classes).to(device)
        self.flat = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1).to(device)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        x = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in outputs[2]]), 0), 0, 1)

        x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)

        x = self.dropout(x)
        x = self.flat(x)
        x = self.dropout(x)
        x = self.fc(x)
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
