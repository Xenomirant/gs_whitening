from transformers import TrainerCallback
import wandb

class LayerLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Log all metrics that start with 'layer_loss_'
            layer_metrics = {k: v for k, v in logs.items() if k.startswith('layer_loss_')}
            if layer_metrics:
                wandb.log(layer_metrics, step=state.global_step)

# In your training script, add the callback to the Trainer:
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[LayerLossCallback()],  # Add the callback here
    # ... other arguments ...
) 