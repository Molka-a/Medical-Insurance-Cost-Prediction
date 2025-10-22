
# ![Medical Insurance Cost Prediction](https://img.shields.io/badge/Project-Medical%20Insurance%20Cost%20Prediction-blue)

## 📌 Project Overview

This project predicts **medical insurance charges** based on personal characteristics using **Linear Regression**.

* **Goal:** Help insurance companies estimate premiums and assist individuals in understanding potential medical costs.
* **Target Audience:** Insurance providers, healthcare analysts, and data science enthusiasts.

---

## 📊 Dataset

Dataset is from [Kaggle: Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance).

**Columns:**

| Feature  | Description                     |
| -------- | ------------------------------- |
| age      | Age of the individual           |
| sex      | Gender (male/female)            |
| bmi      | Body Mass Index                 |
| children | Number of children/dependents   |
| smoker   | Smoker status (yes/no)          |
| region   | Residential region              |
| charges  | Medical insurance cost (target) |

**Dataset Size:** 1338 rows × 7 columns

---

## 🛠 Tools & Libraries

* **Python**
* **NumPy, Pandas** – Data handling
* **Matplotlib, Seaborn** – Data visualization
* **Scikit-learn** – Machine learning

---

## 🧹 Data Preprocessing

* Checked for missing values (none).

* Encoded categorical columns:

  * `sex`: male → 0, female → 1
  * `smoker`: yes → 0, no → 1
  * `region`: southeast → 0, southwest → 1, northeast → 2, northwest → 3

* Split features and target:

```python
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']
```

---

## 🤖 Model Training

**Algorithm:** Linear Regression
**Train/Test Split:** 80% train, 20% test

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
```

**Performance Metrics:**

* R² on training set: **0.7515**
* R² on test set: **0.7447**

---

## 🖥 Predictive System Example

```python
# Input: age, sex, bmi, children, smoker, region
input_data = (31, 1, 25.74, 0, 1, 0)
input_data_as_numpy_array = np.asarray(input_data).reshape(1,-1)

prediction = regressor.predict(input_data_as_numpy_array)
print("Predicted Insurance Cost: USD", prediction[0])
```

**Output:**

```
Predicted Insurance Cost: USD 3760.08
```

---

## 📈 Conclusion

* Linear Regression predicts insurance charges with ~75% accuracy.
* Useful for both individuals and insurance providers to anticipate costs.
* **Future Work:** Test advanced regression models like Random Forest or XGBoost for improved accuracy.

---

## 🔗 References

* Kaggle Dataset: [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)

