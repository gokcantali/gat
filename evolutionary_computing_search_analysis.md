# Evolutinary Computing Search Analysis

## Analysis

### First run

```python
    config_ranges = {
        "optimizers": [torch.optim.AdamW],
        "lr": [0.039, 0.04, 0.041, 0.042, 0.0425],
        "weight_decay": (0.00046, 0.00052),
        "epochs": [30, 40],
        "patience": [5],
        "hidden_dim": [28, 30, 32, 34, 36],
        "dropout": [0.35, 0.375, 0.4, 0.425, 0.45]
    }
```

Based on the result of the evolutionary search, the best configuration is:
Optimizer: torch.optim.AdamW
Learning Rate: 0.0425
Weight Decay: 0.0004807430799298252
Epochs: 30
Patience: 5
Hidden Dimension: 30
Dropout: 0.425

The composite score of this configuration is 0.9986521026584125

## Data

### First run

```text
Best Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.0425, weight_decay=0.0004807430799298252, epochs=30, patience=5, hidden_dim=30, dropout=0.425)
Composite Score: 0.9986521026584125
```
