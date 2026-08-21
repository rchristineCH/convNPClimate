# Training comparison (fold-averaged best)

| Model | best CRPS (epoch) | best MAE (epoch) | max Pearson | max Spearman |
|---|---|---|---|---|
| NLL-trained | 1.7885 (26) | 2.5128 (26) | 0.7211 | 0.8097 |
| CRPS-trained | 1.8323 (29) | 2.5495 (29) | 0.6994 | 0.7745 |
| WIND-trained | 1.7061 (28) | 2.3504 (19) | 0.7456 | 0.8218 |
| FT-CRPS (fine-tune epochs) | 1.7616 (4) | 2.4397 (4) | 0.6892 | 0.8061 |
| SFC-TP-trained | 1.6197 (28) | 2.2620 (28) | 0.7846 | 0.8287 |

Note: `test NLL` is each run's own training objective on the held-out fold (for the CRPS runs it is `-CRPS`, not a likelihood), so only `test CRPS`, MAE and the correlations are cross-comparable.

Note: `FT-CRPS` reports only its 10 fine-tune epochs, which start from the already-converged `NLL` model — its epoch numbers count epochs after the warm start and are not comparable to the from-scratch runs. Compare it on the held-out physical metrics instead.
