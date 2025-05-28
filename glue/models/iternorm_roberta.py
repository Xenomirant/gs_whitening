import regex as re
import torch
from torch import nn as nn
from transformers import RobertaModel

from models.utils import set_layer, singular_norm
from models.layers.whitening import WhiteningSing2dIterNorm, WhiteningMatrixSign2dIterNorm
from typing import Literal


whitening_layer_type = {
        "matrix_sign": WhiteningMatrixSign2dIterNorm, 
        "matrix_root": WhiteningSing2dIterNorm,
                            }

class IterNormRobertaClassifier(nn.Module):

    def __init__(self, n_classes, cls_dropout=0.1, 
        iteration_type: Literal["matrix_sign", "matrix_root"] = "matrix_sign", 
        num_iterations=4, use_running_stats_train=True,
        use_batch_whitening=False, use_only_running_stats_eval=False,
        whitening_affine=True, log_steps_eff_rank=10):
        super().__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-base")

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
                    
                    print(f"Changling layer: {name}")
                    set_layer(self.roberta, name, wh_layer)


        self.eff_ranks = {}
        self.log_every = log_steps_eff_rank
        self.log_step=0
        self._eff_ranks = {}
        self._register_eff_rank_hooks()

        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(768, n_classes))
        
        if n_classes == 1:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        for name, parameter in self.named_parameters():
            parameter.requires_grad=True
    

    def _get_attention_mask_hook(self,):
        def forward_hook(module, input,):
            if hasattr(self, 'attention_mask'):
                module.attention_mask = self.attention_mask
            return None
        return forward_hook


    def _register_eff_rank_hooks(self):
        """Register hooks to calculate and store layer-wise losses"""
        def get_loss_hook(layer_name):
            def hook(module, input, output):
                with torch.no_grad():
                    if self.log_step % self.log_every == 0:
                        current_layer = layer_name
                        
                        if isinstance(input, tuple):
                            input_ = input[0].clone().detach()
                        else:
                            input_ = input.clone().detach()
                        input_eff_rank = (
                            torch.linalg.matrix_norm(input_, 
                                    ord="fro", dim=(-2, -1))**2 / singular_norm(input_)**2
                            ).mean().item()
                        
                        if isinstance(output, tuple):
                            output_ = output[0].clone().detach()
                        else:
                            output_ = output.clone().detach()
                        output_eff_rank = (
                            torch.linalg.matrix_norm(output_, 
                                    ord="fro", dim=(-2, -1))**2 / singular_norm(output_)**2
                            ).mean().item()

                        self._eff_ranks[f"train/input_{current_layer}_eff_rank"] = input_eff_rank
                        self._eff_ranks[f"train/output_{current_layer}_eff_rank"] = output_eff_rank
                return None
            return hook

        # Register hooks for specific layers
        for name, module in self.roberta.named_modules():
            if ("embeddings" in name or re.search(r"encoder\.layer\.[0-9]+\.output", name)) \
                    and (isinstance(module, nn.LayerNorm) or isinstance(module, WhiteningMatrixSign2dIterNorm)):
                print(f"Setting hook on layer:{name}")
                module.register_forward_hook(get_loss_hook(name))

    def forward(self, input_ids, attention_mask, labels=None, **batch):
        
        if self.log_step % self.log_every == 0:
            self._eff_ranks = {}
            self.log_step=0

        self.attention_mask = attention_mask
        
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        pooler = roberta_output[0][:, 0]
        logits = self.classifier(pooler)
        self.log_step+=1
        # self.report_metrics(**self.eff_ranks)
        if labels is not None:
            loss = self.criterion(logits.squeeze(), labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


        