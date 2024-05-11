from dataclasses import dataclass, field
from typing import List


@dataclass
class Metrics:
    epoch_values: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    train_precision: List[float] = field(default_factory=list)
    train_recall: List[float] = field(default_factory=list)
    train_f1: List[float] = field(default_factory=list)
    val_precision: List[float] = field(default_factory=list)
    val_recall: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
