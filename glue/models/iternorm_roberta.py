import regex as re
import torch
from torch import nn as nn
from transformers import RobertaModel

from models.utils import set_layer
from models.layers.whitening import Whitening2dIterNorm


class IterNormRobertaClassifier(nn.Module):

    supports_report_metrics: bool = True

    def __init__(self, n_classes, cls_dropout=0.1,
        norm_iterations=4, use_running_stats_train=True,
        use_batch_whitening=False, use_only_running_stats_eval=False,
        log_steps_eff_rank=10):
        super().__init__()
        
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        for name, module in self.roberta.named_modules():
            if re.search("encoder\.layer\.[0-9]+\.output", name):
                if isinstance(module, nn.LayerNorm):
                    emb_dim = module.weight.shape[0]
                    weight, bias = module.weight.data, module.bias.data

                    wh_layer = Whitening2dIterNorm(num_features=emb_dim, 
                                iterations=norm_iterations, use_batch_whitening=use_batch_whitening,
                                use_running_stats_train=use_running_stats_train,
                                use_only_running_stats_eval=use_only_running_stats_eval
                                )
                    wh_layer.weight.data, wh_layer.bias.data = weight.clone(), bias.clone()
                    print(f"Changling layer: {name}")
                    set_layer(self.roberta, name, wh_layer)


        self.eff_ranks = {}
        self.log_every = log_steps_eff_rank
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
        
    def _register_eff_rank_hooks(self):
        """Register hooks to calculate and store layer-wise losses"""
        def get_loss_hook(layer_name):
            def hook(module, input, output):
                if self.log_step % self.log_every == 0:
                    if isinstance(output, tuple):
                        output = output[0].clone().detach()  # Handle cases where output is a tuple
                    self.eff_ranks[f"train/{layer_name}_eff_rank"] = (
                        torch.linalg.matrix_norm(output, ord="fro", dim=(-2, -1)) / torch.linalg.matrix_norm(output, ord=2, dim=(-2, -1))
                        ).mean().item()
                return None
            return hook

        # Register hooks for specific layers
        for name, module in self.roberta.named_modules():
            if name == "embeddings" or re.search("encoder\.layer\.[0-9]+\.output$", name):
                print(f"Setting hook on layer:{name}")
                module.register_forward_hook(get_loss_hook(name))

    def forward(self, input_ids, attention_mask, labels=None, **batch):
        
        if self.log_step % self.log_every == 0:
            self.eff_ranks = {}
            self.log_step=0
        
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        pooler = roberta_output[0][:, 0]
        logits = self.classifier(pooler)
        self.log_step+=1
        # self.report_metrics(**self.eff_ranks)
        if labels is not None:
            loss = self.criterion(logits.squeeze(), labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


        