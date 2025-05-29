import regex as re
import torch
from torch import nn as nn
from transformers import RobertaModel

from models.utils import set_layer, singular_norm, trace_loss
from models.layers.whitening import WhiteningSing2dIterNorm, WhiteningMatrixSign2dIterNorm, WhiteningTrace2dIterNorm
from models.layers.roberta_abc import ABCRobertaClassifier
from typing import Literal


whitening_layer_type = {
        "matrix_sign": WhiteningMatrixSign2dIterNorm, 
        "matrix_root": WhiteningSing2dIterNorm,
        "matrix_trace": WhiteningTrace2dIterNorm
                            }

class IterNormRobertaClassifier(ABCRobertaClassifier):

    def __init__(self, n_classes, cls_dropout=0.1, 
        iteration_type: Literal["matrix_sign", "matrix_root"] = "matrix_sign", 
        num_iterations=4, use_running_stats_train=True,
        use_batch_whitening=False, use_only_running_stats_eval=False,
        whitening_affine=True, 
        remove_biases=False,
        use_trace_loss=False,
        log_steps_eff_rank=10):
        super().__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.trace_loss = torch.tensor([0], requires_grad=True, device=self.roberta.device)

        for name, module in self.roberta.named_modules():
            if re.search(r"encoder\.layer\.[0-9]+\.output", name):
                if isinstance(module, nn.LayerNorm):
                    num_features = self.roberta.config.hidden_size
                    weight, bias = module.weight.data, module.bias.data

                    wh_layer = whitening_layer_type[iteration_type](num_features=num_features, 
                                iterations=num_iterations, use_batch_whitening=use_batch_whitening,
                                use_running_stats_train=use_running_stats_train,
                                use_only_running_stats_eval=use_only_running_stats_eval,
                                affine=whitening_affine
                                )
                    
                    if whitening_affine:
                        wh_layer.weight.data, wh_layer.bias.data = weight.clone(), bias.clone()
                    
                    wh_layer.register_forward_pre_hook(self._get_attention_mask_hook())

                    if use_trace_loss:
                        wh_layer.register_forward_hook(self._get_trace_loss_hook())
                    
                    print(f"Changling layer: {name}")
                    set_layer(self.roberta, name, wh_layer)

        if remove_biases:
            self.remove_biases()

        self.eff_ranks = {}
        self.log_every = log_steps_eff_rank
        self.log_step=0
        self._eff_ranks = {}
        self._register_eff_rank_hooks()

        self.classifier = nn.Sequential(
            nn.Linear(768, 768, bias=False if remove_biases else True),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(768, n_classes, bias=False if remove_biases else True))
        
        if n_classes == 1:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        for name, parameter in self.named_parameters():
            parameter.requires_grad=True


    def forward(self, input_ids, attention_mask, labels=None, **batch):
        
        if self.log_step % self.log_every == 0:
            self._eff_ranks = {}
            self.log_step=0

        self.attention_mask = attention_mask
        
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        pooler = roberta_output[0][:, 0]
        logits = self.classifier(pooler)
        # self.report_metrics(**self.eff_ranks)
        self.log_step+=1
        if labels is not None:
            loss = self.criterion(logits.squeeze(), labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


        