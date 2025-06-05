import regex as re
import torch
from torch import nn as nn
from transformers import RobertaModel
from peft import LoraConfig, get_peft_model

from models.utils import set_layer, singular_norm, trace_loss
from models.layers.whitening import WhiteningSing2dIterNorm, WhiteningMatrixSign2dIterNorm, WhiteningTrace2dIterNorm
from models.layers.roberta_abc import ABCRobertaClassifier

whitening_layer_type = {
        "matrix_sign": WhiteningMatrixSign2dIterNorm, 
        "matrix_root": WhiteningSing2dIterNorm,
        "matrix_trace": WhiteningTrace2dIterNorm
                            }

class LoraRobertaClassifier(ABCRobertaClassifier):

    def __init__(self, n_classes, r, cls_dropout=0.1, lora_dropout=0.0, 
                 peft_bias='none', use_dora=False, use_whitening=False, whitening_params = None,
                 remove_biases=False, use_trace_loss=True, trace_loss_trade_off = 0.01, whiten_last_layer=False,
                 log_steps_eff_rank=10):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.trace_loss_trade_off=trace_loss_trade_off

        config = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=["query", "value", "key", "output.dense", "intermediate.dense"],
            lora_dropout=lora_dropout,
            bias=peft_bias,
            modules_to_save=[],
            use_dora=use_dora,
        )

        self.roberta = get_peft_model(self.roberta, config)
        
        whitening_re = r"encoder\.layer\.[0-9]+\.output"
        if whiten_last_layer:
            whitening_re = r"encoder\.layer\.11\.output"

        if use_whitening:
            for name, module in self.roberta.named_modules():
                if re.search(whitening_re, name):
                    if isinstance(module, nn.LayerNorm):
                        num_features = self.roberta.config.hidden_size
                        weight, bias = module.weight.data, module.bias.data

                        if whitening_params is None:
                            whitening_params = {}
                        
                        iteration_type = whitening_params.get("iteration_type", "matrix_root")
                        whitening_params["num_features"] = num_features
                        wh_layer = whitening_layer_type[iteration_type](**whitening_params)
                        
                        if whitening_params.get("affine", False):
                            wh_layer.weight.data, wh_layer.bias.data = weight.clone(), bias.clone()
                        
                        wh_layer.register_forward_pre_hook(self._get_attention_mask_hook())
                        
                        if use_trace_loss:
                            wh_layer.register_forward_hook(self._get_trace_loss_hook())
                        
                        print(f"Changling layer: {name}")
                        set_layer(self.roberta, name, wh_layer)

        if remove_biases:
            self.remove_biases()

        self.log_every=log_steps_eff_rank
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
        

    def forward(self, input_ids, attention_mask, labels=None, **batch):
        
        if self.log_step % self.log_every == 0:
            self._eff_ranks = {}
            self.log_step=0

        factory_kwargs = {"device": input_ids.device, "dtype": torch.float32}
        self.attention_mask = attention_mask
        self.trace_loss = torch.tensor(0.0, requires_grad=True, **factory_kwargs)
        
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        pooler = roberta_output[0][:, 0]
        logits = self.classifier(pooler)
        self.log_step += 1
        # self.report_metrics(**self.eff_ranks)
        if labels is not None:
            loss = self.criterion(logits.squeeze(), labels)
            loss += self.trace_loss*self.trace_loss_trade_off
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


        