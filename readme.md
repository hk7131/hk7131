# 🚆 Rake Examination Time Prediction

## 📌 Project Overview

This project focuses on predicting the **rake examination lifecycle time** using Machine Learning techniques.

The project uses railway rake data collected from **New Katni Junction (NKJ)** and analyzes operational timestamps to identify patterns affecting the time taken during different stages of the rake examination process.

The primary goal is to build regression models that can predict examination-related time durations and help improve railway operational efficiency.

---

## 🎯 Objectives

* Analyze railway rake operational data.
* Perform data cleaning and preprocessing.
* Convert timestamp information into meaningful numerical features.
* Calculate time durations between different operational events.
* Encode categorical variables using appropriate techniques.
* Analyze monthly trends in rake examination time.
* Train and compare different regression models.
* Evaluate model performance using MAE, RMSE, and R².

---

## 📊 Dataset

The dataset contains railway rake and examination-related information.

Important features include:

| Feature    | Description                |
| ---------- | -------------------------- |
| `RAKEID`   | Unique rake identification |
| `RAKETYPE` | Type of rake               |
| `EXAMSTTN` | Examination station        |
| `ARVLTIME` | Arrival time               |
| `PLCTTIME` | Placement time             |
| `RELSTIME` | Release time               |
| `DPRTTIME` | Departure time             |
| `SNAME`    | Station name               |
| `ZONE`     | Railway zone               |
| `MONTH`    | Month of operation         |

The project primarily focuses on predicting time-related parameters derived from these operational timestamps.

---

## 🔧 Data Preprocessing

The following preprocessing techniques were applied:

### 1. Datetime Conversion

Operational timestamp columns were converted into Pandas datetime format.

```python
df['PLCTTIME'] = pd.to_datetime(df['PLCTTIME'])
df['RELSTIME'] = pd.to_datetime(df['RELSTIME'])
```

### 2. Duration Calculation

Time differences were calculated between operational events.

For example:

```python
df['PL_TO_RL'] = (
    df['RELSTIME'] - df['PLCTTIME']
).dt.total_seconds()
```

The duration is stored in **seconds** so that it can be directly used for regression.

### 3. Categorical Encoding

Categorical variables such as:

* `EXAMSTTN`
* `SNAME`
* `ZONE`
* `MONTH`

were converted into numerical representations using encoding techniques.

`MONTH` was converted using **one-hot encoding** so that the model does not incorrectly assume a linear relationship between months.

Example:

```python
X = pd.get_dummies(
    X,
    columns=['MONTH'],
    drop_first=True
)
```

---

## 🤖 Machine Learning Models

Different regression algorithms were explored, including:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

The models were trained using a train-test split.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## 📈 Gradient Boosting Results

One of the evaluated models was `GradientBoostingRegressor`.

```python
model_gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
```

### Performance

| Metric |            Result |
| ------ | ----------------: |
| MAE    |  5,802.91 seconds |
| MAE    |     96.72 minutes |
| RMSE   | 12,275.03 seconds |
| RMSE   |    204.58 minutes |
| R²     |             0.727 |

The R² score of approximately **0.727** indicates that the model explains around **72.7% of the variation** in the target variable on the test set.

---

## 📅 Month-wise Analysis

Monthly analysis is performed to understand whether model predictions follow the actual variation in examination time across different months.

The analysis compares:

* Actual average examination time
* Predicted average examination time

This helps identify months where the model performs well and months where prediction errors are relatively higher.

---

## 📉 Evaluation Metrics

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

Lower MAE indicates better performance.

### Root Mean Squared Error — RMSE

Penalizes larger prediction errors more strongly than MAE.

Lower RMSE indicates better performance.

### R² Score

Measures how much variation in the target variable is explained by the model.

A value closer to **1** indicates better explanatory performance.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* Git & GitHub

---

## 📁 Project Structure

```text
Rake-Examination-Prediction/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── rake_prediction.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 🚀 Future Improvements

* Compare additional regression algorithms.
* Perform hyperparameter tuning.
* Use time-series based validation.
* Analyze rake-type-specific performance.
* Investigate outliers causing high RMSE.
* Perform feature importance analysis.
* Build a prediction interface for railway operational users.
* Test the model on future/unseen railway data.

---

## 👨‍💻 Author

**Harsh Kaushal**

B.Tech — Electronics & Communication Engineering


