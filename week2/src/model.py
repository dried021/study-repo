from torch import nn
from transformers import BertModel, BartModel

# 또는 from transformers import BertForSequenceClassification, BartForSequenceClassification 사용
class SentimentClassifier(nn.Module):
    def __init__(self, num_labels, model):
        super(SentimentClassifier, self).__init__()
        self.model_type = model

        if model == 'bert':
            self.pretrained_layer = BertModel.from_pretrained('google-bert/bert-base-uncased')
        else:
            #Bart
            self.pretrained_layer = BartModel.from_pretrained('facebook/bart-base')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.pretrained_layer.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # 구버전
        # _, pooled_output = self.pretrained_layer(
        #     input_ids = input_ids,
        #     attention_mask = attention_mask
        # )
        # output = self.drop(pooled_output)

        if self.model_type == 'bert':
            outputs = self.pretrained_layer(
                input_ids = input_ids,
                attention_mask = attention_mask
            )
            pooled_output = outputs.pooler_output
        else:
            #BART
            outputs = self.pretrained_layer(
                input_ids = input_ids,
                attention_mask = attention_mask
            )
            # BART에서는 pooling layer 출력의 첫 번째 토큰이 전체 문장을 대표
            pooled_output = outputs.last_hidden_state[:, 0, :]

        output = self.drop(pooled_output)
        return self.out(output)


def initialize_model(model, num_labels):
    return SentimentClassifier(num_labels, model)
