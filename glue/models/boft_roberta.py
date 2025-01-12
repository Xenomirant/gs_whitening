from torch import nn as nn
from transformers import RobertaModel
from peft import BOFTConfig, get_peft_model

class BOFTRobertaClassifier(nn.Module):
    def __init__(self, n_classes, boft_block_size, boft_n_butterfly_factor, cls_dropout=0.1, boft_dropout=0.0, bias='none'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        config = BOFTConfig(
            boft_block_size = boft_block_size,
            boft_n_butterfly_factor=boft_n_butterfly_factor,
            target_modules=["query", "value", "key", "output.dense", "intermediate.dense"],
            boft_dropout=boft_dropout,
            bias=bias,
            modules_to_save=[]
        )

        self.roberta = get_peft_model(self.roberta, config)

        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(768, n_classes))
        
        self.criterion = nn.CrossEntropyLoss()

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


        