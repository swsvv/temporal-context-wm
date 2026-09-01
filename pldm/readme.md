## Two-Room Environment

### PLDM + Temporal Context Regularization

```bash
python -m pldm.train --configs pldm/configs/wall/context/seqlen17_3M_modified.yaml
```

### PLDM Baseline

```bash
python -m pldm.train --configs pldm/configs/wall/icml/seqlen17_3M.yaml
```

Full list of configs can be found in `configs/wall/`. YAML files override each other if they share values, with the last element in the list overriding last. The `--values` flag allows modifying loaded configs:

```bash
python -m pldm.train --configs pldm/configs/wall/context/seqlen17_3M_modified.yaml \
  --values base_lr=0.01 data.offline_wall_config.batch_size=128
```

## Hyperparameter Tuning

Key hyperparameters for the context embedding objective (`contextemb` in config):
- `context_objective_coeff`: weight for the context reconstruction loss
- `context_t_loss_coeff`: weight for the temporal consistency loss
- `context_window_k`: context radius (window size = 2k+1)

The PLDM hyperparameters ($\alpha, \beta, \lambda, \delta, \omega$) should be tuned for any new environment or significantly different data distributions.
