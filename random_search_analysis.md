# Random Search Analysis

## Analysis

### First run

```python
    config_ranges = {
        "optimizers": [torch.optim.AdamW],
        "lr": [0.03, 0.05],
        "weight_decay": (0.0004, 0.0006),
        "epochs": [20, 30, 40],
        "patience": [3, 5, 7],
        "hidden_dim": [25, 32, 39],
        "dropout": [0.3, 0.5]
    }
```

Based on the first run learning rates around 0.03-0.05 perform well with weight
decay around 0.0004-0.0006 and a hidden dimension of 25-39. Dropout rates around
0.3-0.5 yield good results.

### Second run

For the second round the values were adjusted based on the first run.
The epoch values and the patience was based on the best performing values from the
first run while for the others a narrower range was chosen for the second run.

```python
    config_ranges = {
        "optimizers": [torch.optim.AdamW],
        "lr": [0.0325, 0.035, 0.0375, 0.04, 0.0425],
        "weight_decay": (0.00045, 0.00055),
        "epochs": [30, 40],
        "patience": [5],
        "hidden_dim": [28, 30, 32, 34],
        "dropout": [0.35, 0.375, 0.4, 0.425, 0.45]
    }
```

The second run shows that the model performs well with a learning rate of 0.04-0.0425,
weight decay of 0.00046-0.00052, epochs of 30-40, patience of 5, hidden dimensions of 28-34
and dropout rates of 0.35-0.45.

So some reasonable values for the hyperparameters are:
- Optimizer: AdamW
- Learning rate: 0.04
- Weight decay: 0.00048
- Epochs: 30
- Patience: 5
- Hidden dimension: 32
- Dropout: 0.4

## Data

### First run

```text
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.044326529610909896, weight_decay=0.00046346341490267456, epochs=30, patience=7, hidden_dim=25, dropout=0.4130287954211385), Avg Val Accuracy: 0.9982944542253521, Avg Val Loss: 0.008541715331375599, Avg Train Loss: 0.06093702092766762, Avg Val Precision: 0.9982945895456502, Avg Val Recall: 0.9982944542253521, Avg Val F1: 0.998294322718108, Composite Score: 0.9982944551786157
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.03930418909987151, weight_decay=0.00047415912491026864, epochs=30, patience=3, hidden_dim=32, dropout=0.3735517027569435), Avg Val Accuracy: 0.9829445422535211, Avg Val Loss: 0.04897623509168625, Avg Train Loss: 0.10765167325735092, Avg Val Precision: 0.9835589130813182, Avg Val Recall: 0.9829445422535211, Avg Val F1: 0.982988041836103, Composite Score: 0.9831090098561159
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.049472198547088155, weight_decay=0.0004069471781639031, epochs=20, patience=7, hidden_dim=39, dropout=0.3725015795990687), Avg Val Accuracy: 0.9978543133802817, Avg Val Loss: 0.00946822389960289, Avg Train Loss: 0.08171933889389038, Avg Val Precision: 0.9978586669372091, Avg Val Recall: 0.9978543133802817, Avg Val F1: 0.9978536882826533, Composite Score: 0.9978552454951064
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.04355236040109688, weight_decay=0.0004623375254206231, epochs=20, patience=5, hidden_dim=32, dropout=0.4342924685317484), Avg Val Accuracy: 0.9967539612676056, Avg Val Loss: 0.016548000276088715, Avg Train Loss: 0.08315011113882065, Avg Val Precision: 0.9967705620952194, Avg Val Recall: 0.9967539612676056, Avg Val F1: 0.996752147978835, Composite Score: 0.9967576581523163
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.03620388742267514, weight_decay=0.0005526375867213268, epochs=30, patience=5, hidden_dim=32, dropout=0.3946716881470342), Avg Val Accuracy: 0.9984595070422535, Avg Val Loss: 0.007605178281664848, Avg Train Loss: 0.058811768889427185, Avg Val Precision: 0.9984609884008756, Avg Val Recall: 0.9984595070422535, Avg Val F1: 0.9984592388694696, Composite Score: 0.998459810338713
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.030147187162968048, weight_decay=0.000520821212426894, epochs=30, patience=7, hidden_dim=25, dropout=0.3571604524766814), Avg Val Accuracy: 0.9961762764084507, Avg Val Loss: 0.014178788289427757, Avg Train Loss: 0.054821841418743134, Avg Val Precision: 0.9961945136247965, Avg Val Recall: 0.9961762764084507, Avg Val F1: 0.9961740065762242, Composite Score: 0.9961802682544806
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.030051228457258387, weight_decay=0.0005807073291879146, epochs=40, patience=5, hidden_dim=32, dropout=0.4185076694788553), Avg Val Accuracy: 0.998597051056338, Avg Val Loss: 0.005447684321552515, Avg Train Loss: 0.05691448226571083, Avg Val Precision: 0.998597270980173, Avg Val Recall: 0.998597051056338, Avg Val F1: 0.9985969361003731, Composite Score: 0.9985970772983056
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.046687470449830536, weight_decay=0.0005038467354084938, epochs=30, patience=5, hidden_dim=32, dropout=0.32621550619171935), Avg Val Accuracy: 0.997771786971831, Avg Val Loss: 0.009649967774748802, Avg Train Loss: 0.04684783145785332, Avg Val Precision: 0.9977759284831454, Avg Val Recall: 0.997771786971831, Avg Val F1: 0.9977711487596727, Composite Score: 0.99777266279662
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.03724290655133427, weight_decay=0.0004165363246996788, epochs=20, patience=3, hidden_dim=39, dropout=0.3511918119696757), Avg Val Accuracy: 0.9940856073943662, Avg Val Loss: 0.021868454292416573, Avg Train Loss: 0.1249178946018219, Avg Val Precision: 0.9941281461218823, Avg Val Recall: 0.9940856073943662, Avg Val F1: 0.9940897344154057, Composite Score: 0.994097273831505
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.036592329804055954, weight_decay=0.0005417868090354799, epochs=40, patience=5, hidden_dim=25, dropout=0.4047635737332601), Avg Val Accuracy: 0.9605798855633803, Avg Val Loss: 0.08733920007944107, Avg Train Loss: 0.11265852302312851, Avg Val Precision: 0.9638771261729278, Avg Val Recall: 0.9605798855633803, Avg Val F1: 0.9607829989811337, Composite Score: 0.9614549740702055
```

### Second run

```text
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.04, weight_decay=0.00046034892805561566, epochs=40, patience=5, hidden_dim=30, dropout=0.4), Avg Val Accuracy: 0.9909496038732394, Avg Val Loss: 0.03020655643194914, Avg Train Loss: 0.1167909304300944, Avg Val Precision: 0.9912296346978069, Avg Val Recall: 0.9909496038732394, Avg Val F1: 0.9909676481369406, Composite Score: 0.9910241226453066
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.0425, weight_decay=0.0004918287855336769, epochs=30, patience=5, hidden_dim=28, dropout=0.45), Avg Val Accuracy: 0.9968731660798124, Avg Val Loss: 0.01376158402611812, Avg Train Loss: 0.09366147468487422, Avg Val Precision: 0.9968791477413728, Avg Val Recall: 0.9968731660798124, Avg Val F1: 0.9968720660043472, Composite Score: 0.9968743864763362
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.04, weight_decay=0.00047921906493976084, epochs=30, patience=5, hidden_dim=32, dropout=0.375), Avg Val Accuracy: 0.997221610915493, Avg Val Loss: 0.015275694740315279, Avg Train Loss: 0.09826050077875455, Avg Val Precision: 0.997229809514829, Avg Val Recall: 0.997221610915493, Avg Val F1: 0.997220639768074, Composite Score: 0.9972234177784722
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.04, weight_decay=0.00047606161738555534, epochs=40, patience=5, hidden_dim=34, dropout=0.375), Avg Val Accuracy: 0.9965980780516434, Avg Val Loss: 0.01640882467230161, Avg Train Loss: 0.08292840669552486, Avg Val Precision: 0.9966111343891887, Avg Val Recall: 0.9965980780516434, Avg Val F1: 0.9965963357271371, Composite Score: 0.9966009065549031
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.035, weight_decay=0.0004928851308649389, epochs=40, patience=5, hidden_dim=30, dropout=0.35), Avg Val Accuracy: 0.9963138204225351, Avg Val Loss: 0.01577985504021247, Avg Train Loss: 0.08715225756168365, Avg Val Precision: 0.9963341754592308, Avg Val Recall: 0.9963138204225351, Avg Val F1: 0.9963114669609704, Composite Score: 0.9963183208163178
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.0325, weight_decay=0.0005030885277872884, epochs=30, patience=5, hidden_dim=30, dropout=0.4), Avg Val Accuracy: 0.99662558685446, Avg Val Loss: 0.01294582181920608, Avg Train Loss: 0.08388857046763103, Avg Val Precision: 0.9966388977354762, Avg Val Recall: 0.99662558685446, Avg Val F1: 0.9966237366568692, Composite Score: 0.9966284520253164
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.0425, weight_decay=0.0005210011001373517, epochs=30, patience=5, hidden_dim=34, dropout=0.45), Avg Val Accuracy: 0.9970748973004695, Avg Val Loss: 0.014171921648085117, Avg Train Loss: 0.11316403249899547, Avg Val Precision: 0.9970831776809593, Avg Val Recall: 0.9970748973004695, Avg Val F1: 0.9970736596656534, Composite Score: 0.9970766579868879
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.0425, weight_decay=0.00048805320968436114, epochs=40, patience=5, hidden_dim=30, dropout=0.375), Avg Val Accuracy: 0.9970657276995305, Avg Val Loss: 0.01569234486669302, Avg Train Loss: 0.0877097025513649, Avg Val Precision: 0.9970766380458135, Avg Val Recall: 0.9970657276995305, Avg Val F1: 0.9970643363142854, Composite Score: 0.9970681074397899
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.035, weight_decay=0.0005207526220640411, epochs=30, patience=5, hidden_dim=30, dropout=0.35), Avg Val Accuracy: 0.9968364876760564, Avg Val Loss: 0.013810082649191221, Avg Train Loss: 0.0767277938624223, Avg Val Precision: 0.9968413760866405, Avg Val Recall: 0.9968364876760564, Avg Val F1: 0.9968360928210896, Composite Score: 0.9968376110649607
Config: Config(optimizer=<class 'torch.optim.adamw.AdamW'>, lr=0.0425, weight_decay=0.000483988862998799, epochs=40, patience=5, hidden_dim=28, dropout=0.45), Avg Val Accuracy: 0.9971390845070421, Avg Val Loss: 0.014720745074252287, Avg Train Loss: 0.11642914265394211, Avg Val Precision: 0.9971486251594163, Avg Val Recall: 0.9971390845070421, Avg Val F1: 0.9971378659802514, Composite Score: 0.997141165038438
```
