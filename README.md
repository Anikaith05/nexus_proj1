# Machine Learning Algorithms from Scratch

This project contains Python implementations of three fundamental machine learning algorithms built from scratch using only NumPy for the core logic.

* **K-Nearest Neighbors (KNN):** A non-parametric algorithm for classification.
* **Linear Regression:** A statistical model for regression, implemented using gradient descent.
* **Logistic Regression:** A model for binary classification, also implemented using gradient descent.

## Project Structure
## Project Structure

* `knn.py`: Class implementation for K-Nearest Neighbors
* `linear_regression.py`: Class implementation for Linear Regression
* `logistic_regression.py`: Class implementation for Logistic Regression
* `test-knn.py`: Training and evaluation script for KNN
* `test-linear-regression.py`: Training script for Linear Regression
* `test-logistic-regression.py`: Training script for Logistic Regression
* `knn_model.pkl`: Saved/serialized KNN model
* `linear_regression_model.pkl`: Saved/serialized Linear Regression model
* `logistic_regression_model.pkl`: Saved/serialized Logistic Regression model
* `requirements.txt`: Project dependencies
* `README.md`: This file
## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install the required dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can train each model by running its corresponding `test-` script. These scripts use sample datasets from `scikit-learn` to train the models and then save the trained models to `.pkl` files using `pickle`.

**example-Train KNN Model:**
```bash
python test-knn.py
