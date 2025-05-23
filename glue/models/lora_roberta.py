import regex as re
import torch
from torch import nn as nn
from transformers import RobertaModel
from peft import LoraConfig, get_peft_model

class LoraRobertaClassifier(nn.Module):

    supports_report_metrics: bool = True

    def __init__(self, n_classes, r, cls_dropout=0.1, lora_dropout=0.0, 
                 bias='none', use_dora=False,
                 log_steps_eff_rank=10):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        config = LoraConfig(
            r=r,
            lora_alpha=r,
            target_modules=["query", "value", "key", "output.dense", "intermediate.dense"],
            lora_dropout=lora_dropout,
            bias=bias,
            modules_to_save=[],
            use_dora=use_dora
        )

        self.roberta = get_peft_model(self.roberta, config)
        self.eff_ranks = {}
        self.log_every=log_steps_eff_rank
        self.log_step=0
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
                    layer_name = layer_name.split("base_model.model")[-1]
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
        self.log_step += 1
        # self.report_metrics(**self.eff_ranks)
        if labels is not None:
            loss = self.criterion(logits.squeeze(), labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


        