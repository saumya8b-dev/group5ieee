# IEEE Project – Group 5: Neural Network Regression

A **neural network regression model** that predicts an output \(f\) from three input variables \(x\), \(y\), and \(z\), using TensorFlow/Keras.

The model is trained on a dataset of 30,000 rows and uses a derived feature to capture the underlying nonlinear relationship between the inputs and the output.

---

## 1. Project Files

| File | Description |
|---|---|
| `ieeeproject.ipynb` | Complete Jupyter notebook containing data exploration, model training, evaluation, and visualizations. |
| `data_group5.csv` | Dataset containing 30,000 rows with columns `sr_no`, `x`, `y`, `z`, and `f`. |
| `model_group5.keras` | Trained neural network model, including its architecture and weights. |
| `scaler_group5.npz` | Training-set means and standard deviations used for feature scaling. |

---

## 2. Project Workflow

### 2.1 Data Loading and Cleaning

- Load the dataset using Pandas.
- Remove the `sr_no` column, as it is not a model feature.
- Check for missing values.
- No missing values were found in the dataset.

### 2.2 Exploratory Data Analysis (EDA)

The following visualizations are used to understand the dataset:

- Histograms
- Scatter plots of individual inputs against the output
- Correlation heatmap
- Box plots
- Pair plot

The individual inputs have approximately zero linear correlation with the output. This suggests that the relationship is strongly nonlinear and cannot be adequately described using simple linear regression.

### 2.3 Identifying the Underlying Function

Exploration revealed that the output closely follows:

\[
f \approx \frac{yz}{1+z^2}+0.63+\epsilon
\]

where:

- \(x,y,z\) are the input variables.
- \(0.63\) is an approximately constant offset.
- \(\epsilon\) represents random noise.

The following derived feature is introduced:

\[
\boxed{g=\frac{yz}{1+z^2}}
\]

This feature is added as a fourth input to the neural network to help it learn the underlying relationship more effectively.

### 2.4 Dataset Splitting

The dataset is split in its original row order, without shuffling.

| Dataset | Number of Rows | Percentage |
|---|---:|---:|
| Training | 21,000 | 70% |
| Validation | 4,500 | 15% |
| Testing | 4,500 | 15% |
| **Total** | **30,000** | **100%** |

### 2.5 Feature Scaling

All four input features are standardized using the training dataset's mean and standard deviation.

\[
X_{\text{scaled}}=\frac{X-\mu_{\text{train}}}{\sigma_{\text{train}}}
\]

The same scaling parameters are used for the validation set, test set, and future predictions.

---

## 3. Neural Network Architecture

The model uses a fully connected feedforward neural network.

### Input Features

1. \(x\)
2. \(y\)
3. \(z\)
4. \(g=\frac{yz}{1+z^2}\)

### Architecture

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 4 | — |
| Dense 1 | 256 | ReLU |
| Dense 2 | 256 | ReLU |
| Dense 3 | 128 | ReLU |
| Dense 4 | 64 | ReLU |
| Output | 1 | Linear |

**Total trainable parameters:** 108,289

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | Mean Squared Error (MSE) |
| Batch Size | 128 |
| Maximum Epochs | 400 |
| Learning Rate Reduction Factor | 0.33 |
| Learning Rate Patience | 8 epochs |
| Early Stopping Patience | 30 epochs |
| Best Weights Restoration | Enabled |

The learning rate is reduced by a factor of 0.33 if validation performance does not improve for 8 consecutive epochs.

Early stopping terminates training after 30 epochs without improvement and restores the best-performing model weights.

In the recorded training run, the model stopped at **epoch 96**.

---

## 4. Error Evaluation Criteria

The validation dataset is used to establish the error threshold.

The maximum absolute output value on the validation set is:

\[
M=\max |f|=48.04
\]

Its order of magnitude is:

\[
n=1
\]

The following criteria are used to classify prediction errors:

| Classification | Error Range |
|---|---|
| Good | Error of order \(\leq n-2\), i.e. below 1 |
| Acceptable | Error of order \(n-1\), i.e. from 1 to below 10 |
| Unacceptable | Error of 10 or greater |

Here, the prediction error is calculated as:

\[
E_i=|f_i-\hat f_i|
\]

where \(f_i\) is the actual output and \(\hat f_i\) is the predicted output.

### Test Set Results

| Classification | Number of Predictions | Percentage |
|---|---:|---:|
| Good (Error < 1) | 4,289 | 95.3% |
| Acceptable (1 ≤ Error < 10) | 211 | 4.7% |
| Unacceptable (Error ≥ 10) | 0 | 0% |
| **Total** | **4,500** | **100%** |

**Maximum absolute error on the test set: 1.35**

All test predictions fall within the good or acceptable error categories.

---

## 5. Understanding the Noise Floor

The dataset contains random noise that cannot be predicted from the available input variables.

The estimated noise characteristics are:

- Standard deviation: approximately 0.53
- Maximum noise magnitude: approximately 1.5

Even a model that perfectly learns the underlying function cannot eliminate this random noise.

The theoretical performance expected from the underlying function on the same noisy dataset is approximately:

| Metric | Expected Value |
|---|---:|
| MSE | 0.27–0.28 |
| MAE | 0.43 |
| Maximum Absolute Error | Approximately 1.35 |

The obtained test results are close to this estimated noise limit.

Therefore, the remaining prediction error is primarily attributable to random noise rather than a significant modelling weakness.

---

## 6. Running the Project

### Requirements

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Option 1: Run the Complete Notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `ieeeproject.ipynb`.
3. Upload `data_group5.csv` to the Colab session.
4. Run the notebook from top to bottom.

This performs data exploration, preprocessing, model training, evaluation, and visualization.

### Option 2: Use the Pretrained Model

If you only want to generate predictions without retraining the network, use the saved model and scaler.

```python
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(
    "model_group5.keras",
    compile=False
)

# Load the saved scaling parameters
s = np.load("scaler_group5.npz")

# x_new is a DataFrame containing columns x, y, z

# Generate the derived feature
x_new["g"] = (
    x_new["y"] * x_new["z"]
    / (1 + x_new["z"] ** 2)
)

# Scale the input features
X_scaled = (
    (x_new - s["mean"]) / s["std"]
).values.astype("float32")

# Generate predictions
pred = model.predict(X_scaled)
```

### Important Notes

- The model requires **four input features** in the exact order: `x`, `y`, `z`, `g`.
- The derived feature `g` must be calculated before prediction.
- Inputs must be standardized using the saved training-set parameters.
- Do not fit a new scaler on the prediction data.
- Omitting the derived feature or scaling step will produce incorrect predictions.
- Model (model_group5.keras). A fully connected neural network built with TensorFlow/Keras that predicts the target f from three inputs (x, y, z) plus one engineered feature, g = y·z / (1 + z²). It has hidden layers of 256, 256, 128 and 64 units with ReLU activation and one linear output. It was trained with the Adam optimizer on MSE loss, using a learning-rate reduction on plateau and early stopping, and it stores the weights from the best validation epoch. The data was split by row into 21,000 for training, 4,500 for validation and 4,500 for testing.

Scaler (scaler_group5.npz). The mean and standard deviation of each of the four input columns, calculated on the training rows only. Inputs must be standardized as (input − mean) / std before they go into the model, because the model was trained on standardized values.

Using them together. Build g from the raw y and z, standardize all four columns with the scaler, then pass them to the model.

---

## 7. Conclusion

This project demonstrates the application of a feedforward neural network to a nonlinear regression problem.

Through exploratory data analysis, the underlying mathematical relationship was identified and incorporated into the model through feature engineering.

The trained model achieved:

- **95.3%** of test predictions with absolute error below 1.
- **0** test predictions with absolute error of 10 or greater.
- A maximum absolute test error of **1.35**.

The results approach the estimated noise floor of the dataset, indicating that the model has successfully learned the underlying relationship.

---

## 8. Technologies Used

- **Python** – Programming language
- **TensorFlow/Keras** – Neural network implementation and training
- **NumPy** – Numerical computations
- **Pandas** – Data loading and manipulation
- **Scikit-learn** – Data preprocessing and evaluation
- **Matplotlib & Seaborn** – Data visualization

---

**Project:** IEEE – Group 5  
**Topic:** Neural Network Regression
