---
library_name: peft
license: other
base_model: unsloth/gemma-3-1b-it
tags:
- base_model:adapter:unsloth/gemma-3-1b-it
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: gemma3_formula_lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# gemma3_formula_lora

This model is a fine-tuned version of [unsloth/gemma-3-1b-it](https://huggingface.co/unsloth/gemma-3-1b-it) on the formulas dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.05
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.15.0+cpu
- Datasets 4.0.0
- Tokenizers 0.22.2