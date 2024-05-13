# Bayesian Search Analysis

## Analysis

Based on previous results, the following search space was defined for the
bayesian search:

```python
space = [
    Categorical([torch.optim.AdamW], name="optimizer"),
    Real(0.035, 0.05, name="lr"),
    Real(4.5e-4, 6e-4, name="weight_decay"),
    Integer(25, 35, name="epochs"),
    Integer(3, 7, name="patience"),
    Integer(24, 40, name="hidden_dim"),
    Real(0.4, 0.45, name="dropout")
]
```

Based on the result of the bayesian search, the best configuration is:
Optimizer: torch.optim.AdamW
Learning Rate: 0.03509992469632132
Weight Decay: 0.0006
Epochs: 30
Patience: 3
Hidden Dimension: 24
Dropout: 0.43889497320491244

The composite score of this configuration is 0.9985421382047263

## Data

### First run

```text
Best Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.03509992469632132, weight_decay=0.0006, epochs=30, patience=3, hidden_dim=24, dropout=0.43889497320491244)
Composite Score: 0.9985421382047263
```
