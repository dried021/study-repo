from torch import nn
from transformers import BertModel, BartModel
import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from src.transformer_model import TransformerEncoder


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes=2, **encoder_params):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, **encoder_params)
        self.linear = nn.Linear(encoder_params['d_model'], n_classes)

    def forward(self, input_ids, attention_mask=None):
        # [batch_size, seq_len, d_model]
        enc_out = self.encoder(input_ids)  

        # cls token
        cls_token = enc_out[:, 0, :]     # <sos> 토큰 사용
        logits = self.linear(cls_token)

        # Max pool
        # pooled = enc_out.max(dim=1)[0]
        # logits = self.linear(pooled)
        return logits


# 또는 from transformers import BertForSequenceClassification, BartForSequenceClassification 사용
class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.pretrained_layer = BertModel.from_pretrained('google-bert/bert-base-uncased')
        self.encoder = None
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.pretrained_layer.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # 구버전
        # _, pooled_output = self.pretrained_layer(
        #     input_ids = input_ids,
        #     attention_mask = attention_mask
        # )
        # output = self.drop(pooled_output)

        outputs = self.pretrained_layer(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        # pooler_output은 첫번째 [CLS] 토큰의 마지막 hidden_state
        pooled_output = outputs.pooler_output

        output = self.drop(pooled_output)
        return self.out(output)


def initialize_model(model, num_labels, vocab_size = 10000, encoder_params = None):
    if model in ["bert"]:
        print("model : BERT")
        return BertClassifier(num_labels)
    elif (model == "transformer"):
        print("model : TRANSFORMER ENCODER")
        return TransformerClassifier(vocab_size, num_labels, **encoder_params)

