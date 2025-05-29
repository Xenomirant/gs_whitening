import regex as re
import torch
from torch import nn as nn
from models.utils import singular_norm, trace_loss
from transformers import RobertaModel
from models.layers.roberta_abc import ABCRobertaClassifier


class RobertaClassifier(ABCRobertaClassifier):

    def __init__(self, n_classes, cls_dropout=0.1, 
                 remove_biases=False,
                 log_steps_eff_rank=10):
        super().__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        
        self.log_every = log_steps_eff_rank
        self.log_step=0
        self._eff_ranks = {}
        self._register_eff_rank_hooks()

        if remove_biases:
            self.remove_biases()

        self.classifier = nn.Sequential(
            nn.Linear(768, 768, bias=False if remove_biases else True),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(768, n_classes, bias=False if remove_biases else True))
        
        if n_classes == 1:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            

    def forward(self, input_ids, attention_mask, labels=None, **batch):
        
        if self.log_step % self.log_every == 0:
            self._eff_ranks = {}
            self.log_step=0
        
        
        self.attention_mask = attention_mask

        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        pooler = roberta_output[0][:, 0]
        logits = self.classifier(pooler)
        self.log_step += 1
        # self.report_metrics(**self.eff_ranks)
        if labels is not None:
            loss = self.criterion(logits.squeeze(), labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}