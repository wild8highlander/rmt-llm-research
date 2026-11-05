# Stage 4 — Interpretability bridge

*Run*: `stage4_interpretability_v1` · hypothesis `S4` · protocol v1.1

## Passport

- model: `EleutherAI/pythia-70m`
- config_hash: `af0d05652f5b0f67`, weight_sha256: `ebfa4e2f18696ebd…`
- n_params: 70,426,624
- transformers 5.22.1, torch 2.15.0+cpu

*code*: `fad992198d24 (dirty)`

## Logit lens × BBP spike alignment

| prompt | final P(answer) | r(commit, gap) | p | spike L | commit L |
|---|---|---|---|---|---|
| The capital of France is | 0.002 | -0.28 | 0.836 | 3 | 5 |
| The capital of Japan is | 0.003 | -0.31 | 0.608 | 3 | 5 |
| Two plus two equals | 0.006 | -0.35 | 0.803 | 3 | 5 |
| The author of Romeo and Juliet is | 0.000 | 0.02 | 0.920 | 3 | 5 |
| Water boils at a temperature of 100 de | 0.677 | -0.20 | 0.738 | 2 | 5 |
| The largest planet in the solar system | 0.005 | -0.19 | 0.945 | 3 | 5 |

- Positive r = the layer that commits the answer is the layer with the spectral spike.

## Activation patching × BBP spike alignment

| prompt | P_clean | P_corrupt | r(restore, gap) | p |
|---|---|---|---|---|
| The capital of France is | 0.002 | 0.000 | 0.20 | 0.652 |
| The capital of Japan is | 0.003 | 0.000 | -0.10 | 0.802 |
| Two plus two equals | 0.006 | 0.009 | -0.77 | 0.057 |
| The author of Romeo and Juliet is | 0.000 | 0.000 | 0.60 | 0.241 |
| Water boils at a temperature of 100 de | 0.677 | 0.762 | -0.55 | 0.270 |
| The largest planet in the solar system | 0.005 | 0.005 | 0.55 | 0.262 |

## Heavy-tail alpha (Martin & Mahoney bands)

| matrix | alpha | xmin | KS | lambda_max | band |
|---|---|---|---|---|---|
| gpt_neox.layers.0.mlp.dense_h_to_4h.weight | 6.38 | 5.43e-03 | 0.063 | 4.05e-02 | random-like |
| gpt_neox.layers.0.mlp.dense_4h_to_h.weight | 5.42 | 1.84e-03 | 0.084 | 9.01e-03 | weak |
| gpt_neox.layers.1.mlp.dense_h_to_4h.weight | 3.46 | 5.70e-03 | 0.028 | 7.68e-02 | weak |
| gpt_neox.layers.1.mlp.dense_4h_to_h.weight | 3.29 | 2.42e-03 | 0.036 | 4.82e-02 | weak |
| gpt_neox.layers.2.mlp.dense_h_to_4h.weight | 3.83 | 5.45e-03 | 0.043 | 4.05e-01 | weak |
| gpt_neox.layers.2.mlp.dense_4h_to_h.weight | 3.58 | 2.88e-03 | 0.043 | 2.11e-02 | weak |
| gpt_neox.layers.3.mlp.dense_h_to_4h.weight | 3.83 | 5.70e-03 | 0.066 | 3.29e-02 | weak |
| gpt_neox.layers.3.mlp.dense_4h_to_h.weight | 3.04 | 4.07e-03 | 0.052 | 5.83e-02 | weak |
| gpt_neox.layers.4.mlp.dense_h_to_4h.weight | 3.68 | 5.31e-03 | 0.062 | 4.62e-02 | weak |
| gpt_neox.layers.4.mlp.dense_4h_to_h.weight | 3.59 | 5.63e-03 | 0.055 | 4.91e-02 | weak |
| gpt_neox.layers.5.mlp.dense_h_to_4h.weight | 3.14 | 4.78e-03 | 0.029 | 5.20e-02 | weak |
| gpt_neox.layers.5.mlp.dense_4h_to_h.weight | 4.29 | 7.14e-03 | 0.061 | 2.19e-01 | weak |

- alpha(x gap_ratio): {'pearson_r': -0.6855604766098382, 'n_layers': 6, 'matrices_per_layer': {'0': 2, '1': 2, '2': 2, '3': 2, '4': 2, '5': 2}}
