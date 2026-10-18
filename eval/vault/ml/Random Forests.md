# Random Forests

An ensemble of decision trees, each trained on a bootstrap sample of the data (bagging) and, at every split, only allowed to consider a random subset of the features. Predictions are averaged (regression) or voted (classification).

A single deep decision tree overfits badly: it has low bias and enormous variance. Averaging hundreds of de-correlated trees keeps the low bias and cancels most of the variance, which is why random forests overfit far less than individual trees and work well with almost no tuning.

Useful by-products:
- **Out-of-bag error**: each tree never saw ~37% of the rows, so you get a free validation estimate.
- **Feature importance** from impurity decrease or permutation.

Compared with gradient-boosted trees (XGBoost, LightGBM), forests are easier to tune but usually a bit less accurate on tabular benchmarks. Still my default first baseline for tabular data.
