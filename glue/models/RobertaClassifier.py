from torch import nn as nn
from transformers import RobertaModel



class RobertaClassifier(nn.Module):
    def __init__(self, n_classes, cls_dropout=0.1):
        super().__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(768, n_classes))
        
        if n_classes == 1:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        

    def forward(self, input_ids, attention_mask, labels=None, **batch):
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        pooler = roberta_output[0][:, 0]
        logits = self.classifier(pooler)
        if labels is not None:
            loss = self.criterion(logits.squeeze(), labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}