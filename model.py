# VAT_D
# Copyright (c) 2022-present NAVER Corp.
# Apache License v2.0


import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig, BertForMaskedLM

class ClassificationBert(nn.Module):
    def __init__(self, model_version, num_labels=2):
        super(ClassificationBert, self).__init__()
        config = BertConfig.from_pretrained(model_version)
        config.num_labels = num_labels
        self.bert = BertForSequenceClassification.from_pretrained(model_version, config=config)
        
    def forward(self, inputs):
        outputs = self.bert(**inputs)
        return outputs[0]


class LMBert(nn.Module):
    def __init__(self, model_version):
        super(LMBert, self).__init__()
        config = BertConfig.from_pretrained(model_version)
        self.bert = BertForMaskedLM.from_pretrained(model_version, config=config)
        
    def forward(self, inputs):
        outputs = self.bert(**inputs)
        return outputs[0]