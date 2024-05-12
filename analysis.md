# Story of the hyperparameter tuning process

To compare the performance of different hyperparameters, we used a composite score from
accuracy, precision, recall and F1-score. They are normalized,
then weighted equally and summed up.

## Optimizers

First we compared the performance of different optimizers. We used the following optimizers:

```text
Optimizer: torch.optim.adamw.AdamW
Composite Score: 0.9971768496712957

Optimizer: torch.optim.sgd.SGD
Composite Score: 0.9619766054637474

Optimizer: torch.optim.rmsprop.RMSprop
Composite Score: 0.9970125491793945
```

We can see that the AdamW and RMSprop optimizers performed better than the SGD optimizer.
We will use the AdamW optimizer and the RMSprop optimizer for further hyperparameter tuning.

## Learning Rate

The learning rate was set to 0.1, 0.01, 0.001 and 0.0001.
For the AdamW optimizer, the composite score was best with a learning rate of 0.1 while with
the RMSprop optimizer, the composite score was best with a learning rate of 0.0001.

Let's try to find the best learning rate for the AdamW optimizer.
It turns out that it is around 0.03 and 0.05. Smaller makes it worse, higher makes it worse.
It is about 0.998.

RMSprop seems less stable around 0.0001, therefore further experiments will be conducted with
AdamW and a learning rate of 0.04.

## Weight Decay

The weight decay was set to 0.0001, 0.0005, 0.0007, 0.001, 0.005. The result is quite similar
but the model performs best with a weight decay of 0.0005 considering the composite score.

## Epochs

Based on the loss and accuracy curve we could see that the model started overfitting after
around 40 epochs. We choose to look into 20, 30, 40, 50 epochs. The result shows that around 30
epochs is the best according to the composite score.

## Patience

We choose to test the values 3, 5, 7 and 10 for the patience (stopping the training if the validation
loss does not decrease for a certain number of epochs).
With patience 3, the composite score is the best looking at the composite score.

## Hidden dimensions

To look into the hidden layers the values 8, 16, 32, 64 and 128 were chosen. Here the result
was a bit clearer. The composite score was best with a hidden dimension of 32. Possibly a closer
value between 16 and 64 might still improve the model a bit.

## Dropout

The dropout was set to 0.3, 0.4, 0.5, 0.6 and 0.7. The model performed best with a dropout of 0.4.
The result was not very clear though and might not have that a high impact on the model.
