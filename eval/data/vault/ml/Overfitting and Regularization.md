# Overfitting and Regularization

Overfitting = the model memorises the training set, noise included, and does worse on new data. Symptom: training loss keeps dropping while validation loss turns upward.

Bias-variance framing: high-capacity models have low bias but high variance; regularisation trades a little bias for a big drop in variance.

Tools I reach for:
- **L2 / weight decay**: penalise large weights.
- **L1**: pushes weights to exactly zero (sparse models).
- **Dropout** for neural nets.
- **Early stopping**: stop when validation loss stops improving.
- **Data augmentation** and simply getting more data.
- **Ensembles**: averaging many models (bagging, random forests) cancels out their individual variance.

Always keep a held-out test set you never tune on, otherwise you end up overfitting to the validation set through hyperparameter search.
