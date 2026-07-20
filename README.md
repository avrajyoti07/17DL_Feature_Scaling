# Feature Scaling — Why It Matters for Neural Networks

🔗 **Repo:** [github.com/avrajyoti07/17DL_Feature_Scaling](https://github.com/avrajyoti07/17DL_Feature_Scaling)

A direct, numeric before/after demonstration of why feature scaling matters when training a neural network: the exact same architecture, on the exact same data, trained once on raw features and once on standardized features — with dramatically different results. This is the seventeenth project in my Deep Learning starter phase.

## What's in this repo

| File | Description |
|---|---|
| `11feature_scaling.ipynb` | Main notebook — unscaled model, `StandardScaler`, and scaled model for comparison |
| `Social_Network_Ads.csv` | Dataset — 400 records |
| `README.md` | You're here |

## Dataset

The classic **Social Network Ads** dataset — 400 users, predicting whether someone purchased a product after seeing a social media ad.

| Column | Description |
|---|---|
| `Age` | User's age |
| `EstimatedSalary` | User's estimated salary |
| `Purchased` | Target — `1` if they purchased, `0` if not |

(`User ID` and `Gender` are dropped early on — they're not used as features here.)

## What the notebook does

1. **Load and clean** — drop `User ID` and `Gender`, keep `Age`, `EstimatedSalary`, and `Purchased`
2. **Visualize** — scatter plot of `Age` vs. `EstimatedSalary`
3. **Split** into `X` (`Age`, `EstimatedSalary`) / `y` (`Purchased`), then `train_test_split` (80/20)
4. **Model 1 — no scaling**: `Dense(128, activation='relu')` → `Dense(1, activation='sigmoid')`, compiled with `Adam` + `binary_crossentropy`, trained on the **raw, unscaled** features for 100 epochs
5. **Plot validation accuracy** over those 100 epochs — it's extremely unstable
6. **Apply `StandardScaler`** — fit on the training features, then transform both train and test (correctly done here: no fitting on the test set)
7. **Model 2 — with scaling**: the identical architecture and hyperparameters, trained on the **same features after scaling**, for 100 epochs
8. **Plot validation accuracy again** — this time it climbs smoothly and stays high

## Concepts covered

- Why raw, unscaled features (`Age` ~18–60 vs. `EstimatedSalary` ~15,000–150,000) blow up a network's initial weighted sums, leading to enormous, unstable loss values and a model that can't learn effectively
- `StandardScaler` used correctly: fit only on the training set, then applied (`.transform`, not `.fit_transform`) to the test set, so there's no data leakage between them
- Reading a `val_accuracy` curve to see, very concretely, what scaling changes in practice
- Isolating a single variable — everything except the scaling step is identical between the two runs, making the comparison a clean, controlled experiment

## Tech stack

- Python 3
- `pandas`, `numpy`
- `seaborn`, `matplotlib`
- `scikit-learn`
- `tensorflow` / `keras`

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/avrajyoti07/17DL_Feature_Scaling.git
cd 17DL_Feature_Scaling
```

**2. Install dependencies**
```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow
```

**3. Run the notebook**
```bash
jupyter notebook 11feature_scaling.ipynb
```
Run all cells top to bottom.

## Results

| | No scaling (raw features) | With `StandardScaler` |
|---|---|---|
| Epoch 1 loss | **9377.4** | 0.67 |
| Final (epoch 100) train accuracy | 50.6% | **87.5%** |
| Final (epoch 100) train loss | 62.3 | **0.29** |
| Final (epoch 100) val accuracy | 36.25% | **88.75%** |
| Final (epoch 100) val loss | 22.6 | **0.31** |

## Key takeaway

Same architecture, same data, same hyperparameters — the *only* difference between the two runs is whether the two features were put on a comparable scale first. Without scaling, the loss starts in the thousands and the model never really learns (final accuracy barely above chance, at 50.6%/36.25%). With scaling, the loss starts under `1` and the model reaches ~88% accuracy on both train and validation sets. It's hard to find a cleaner demonstration of why feature scaling is a standard, non-optional step before training a neural network on tabular data.

## Next steps

- [ ] Try `MinMaxScaler` instead of `StandardScaler` and compare results
- [ ] Add early stopping (tying back into the [Early Stopping](https://github.com/avrajyoti07/16DL_Early_Stopping) project) to the scaled model
- [ ] Scale only one of the two features at a time to see how much each contributes to the improvement
- [ ] Plot the learned decision boundary for the scaled model, as in earlier projects, to see it visually

## Related projects in this series

- [1. Perceptron Demo](https://github.com/avrajyoti07/1DL_Perceptron_demo)
- [2. Hand Digits Classification](https://github.com/avrajyoti07/2DL_Hand_Digits_Classification)
- [3. Activation Functions](https://github.com/avrajyoti07/3DL_Activation_Function)
- [4. Matrix Basics](https://github.com/avrajyoti07/4DL_Matrix_Basics)
- [5. Loss & Cost Functions](https://github.com/avrajyoti07/5DL_Loss_-_Cost_Function)
- [6. Gradient Descent](https://github.com/avrajyoti07/6DL_Gradient_Discent)
- [7. Perceptron Trick](https://github.com/avrajyoti07/7DL_Perceptron_Trick)
- [8. Hinge Loss Perceptron](https://github.com/avrajyoti07/8DL_Hinge_Loss_Perceptron)
- [9. Customer Churn Prediction (ANN)](https://github.com/avrajyoti07/9DL_Customer_Churn_Prediction)
- [10. MNIST Classification](https://github.com/avrajyoti07/10DL_mnist_classification)
- [11. Graduate Admission Prediction (ANN)](https://github.com/avrajyoti07/11DL_Graduate_Admission_Prediction)
- [12. Back Prop Classification](https://github.com/avrajyoti07/12DL_Back_Prop_Classification)
- [13. Back Prop Regression](https://github.com/avrajyoti07/13DL_Back_Prop_Regression)
- [16. Early Stopping](https://github.com/avrajyoti07/16DL_Early_Stopping)

## Author

Built by [avrajyoti07](https://github.com/avrajyoti07) as part of a Deep Learning learning journey.
Feel free to fork, star, or open an issue if you spot something to improve!

## License

This project is open source under the [MIT License](LICENSE).
