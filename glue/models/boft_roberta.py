import regex as re
import torch
from torch import nn as nn
from transformers import RobertaModel
from peft import BOFTConfig, get_peft_model

from models.utils import set_layer, singular_norm
from models.layers.whitening import WhiteningSing2dIterNorm, WhiteningMatrixSign2dIterNorm

whitening_layer_type = {
        "matrix_sign": WhiteningMatrixSign2dIterNorm, 
        "matrix_root": WhiteningSing2dIterNorm,
                            }

class BOFTRobertaClassifier(nn.Module):

    def __init__(self, n_classes, boft_block_size, boft_n_butterfly_factor, 
            cls_dropout=0.1, boft_dropout=0.0, 
            bias='none', use_whitening=False, whitening_params = None,
            log_steps_eff_rank=10):
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

        if use_whitening:
            for name, module in self.roberta.named_modules():
                if re.search(r"encoder\.layer\.[0-9]+\.output", name):
                    if isinstance(module, nn.LayerNorm):
                        emb_dim = self.roberta.config.hidden_size
                        weight, bias = module.weight.data, module.bias.data

                        if whitening_params is None:
                            whitening_params = {}
                        
                        iteration_type = whitening_params.get("iteration_type", "matrix_sign")
                        whitening_params["num_features"] = emb_dim
                        wh_layer = whitening_layer_type[iteration_type]()
                        
                        if whitening_params.get("whitening_affine", False):
                            wh_layer.weight.data, wh_layer.bias.data = weight.clone(), bias.clone()
                        
                        wh_layer.register_forward_pre_hook(self._get_attention_mask_hook())
                        
                        print(f"Changling layer: {name}")
                        set_layer(self.roberta, name, wh_layer)

        self.log_every = log_steps_eff_rank
        self.log_step=0
        self._eff_ranks = {}
        self._register_eff_rank_hooks()


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
        
    def _register_eff_rank_hooks(self):
        """Register hooks to calculate and store layer-wise losses"""
        def get_loss_hook(layer_name):
            def hook(module, input, output):
                if self.log_step % self.log_every == 0:
                    
                    current_layer = layer_name.split("base_model.model.")[-1]
                    
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
                
                return output
            return hook

        # Register hooks for specific layers
        for name, module in self.roberta.named_modules():
            if ("embeddings" in name or re.search(r"encoder\.layer\.[0-9]+\.output", name)) \
                    and (isinstance(module, nn.LayerNorm)):
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
        # self.report_metrics(**self.eff_ranks)
        self.log_step += 1
        if labels is not None:
            loss = self.criterion(logits.squeeze(), labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


        
