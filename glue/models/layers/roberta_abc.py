import regex as re
import torch
from torch import nn as nn

from models.utils import set_layer, singular_norm, trace_loss
from models.layers.whitening import WhiteningSing2dIterNorm, WhiteningMatrixSign2dIterNorm, WhiteningTrace2dIterNorm
from abc import abstractmethod, ABC


class ABCRobertaClassifier(nn.Module,):
    
    def _get_attention_mask_hook(self,):
        def forward_hook(module, input,):
            if hasattr(self, 'attention_mask'):
                module.attention_mask = self.attention_mask.clone()
            return None
        return forward_hook
    
    def _get_trace_loss_hook(self,):
        def forward_trace_hook(module, input, output):
            if hasattr(self, "trace_loss"):
                self.trace_loss = self.trace_loss + trace_loss(output)
            return None
        return forward_trace_hook
        
    def _register_eff_rank_hooks(self):
        """Register hooks to calculate and store layer-wise losses"""
        def get_loss_hook(layer_name):
            def hook(module, input, output):
                with torch.no_grad():
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
                return None
            return hook

        # Register hooks for specific layers
        for name, module in self.roberta.named_modules():
            if ("embeddings" in name or re.search(r"encoder\.layer\.[0-9]+\.output", name)) \
                    and (isinstance(module, nn.LayerNorm) or \
                        isinstance(module, WhiteningSing2dIterNorm) or \
                        isinstance(module,WhiteningMatrixSign2dIterNorm) or \
                        isinstance(module,WhiteningTrace2dIterNorm)):
                print(f"Setting hook on layer:{name}")
                module.register_forward_hook(get_loss_hook(name))

    def remove_biases(self):
        """Remove biases from all linear layers in the model"""
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.bias = None
                module.bias_requires_grad = False

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError