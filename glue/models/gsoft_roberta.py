from torch import nn as nn
from transformers import RobertaModel
from models.utils import get_layer, set_layer

from models.layers.gsoft import GSOFTLayer

class GSOFTRobertaClassifier(nn.Module):
    def __init__(self, n_classes, cls_dropout=0.1, nblocks=16, do_gsoft=True, orthogonal=True, method='cayley', block_size=None, scale=True):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base")

        for name, module in self.roberta.named_modules():
            if True: # if "query" in name:
              if isinstance(module, nn.Linear):
                qkv_layer = get_layer(self.roberta, name) # Linear
                in_f, out_f = qkv_layer.weight.shape[1], qkv_layer.weight.shape[0]
                if do_gsoft:
                    gsoft_layer = GSOFTLayer(qkv_layer, in_features=in_f, out_features=out_f, nblocks=nblocks, 
                                            orthogonal=orthogonal, method=method, block_size=block_size, scale=scale)
                    print("changing layer", name)
                    set_layer(self.roberta, name, gsoft_layer)

        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(768, n_classes))
        
        if n_classes == 1:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        if do_gsoft:
            # freeze roberta params
            for name, param in self.roberta.named_parameters():
                if "gsoft_" not in name:
                    param.requires_grad = False


    def forward(self, input_ids, attention_mask, labels=None, **batch):
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        pooler = roberta_output[0][:, 0]
        logits = self.classifier(pooler)
        if labels is not None:
            loss = self.criterion(logits.squeeze(), labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}