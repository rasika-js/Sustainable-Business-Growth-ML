import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def generate_dataset(size=100):
    np.random.seed(42)

    energy = np.random.randint(40, 100, size)
    waste = np.random.randint(40, 100, size)
    csr = np.random.randint(30, 100, size)
    employee = np.random.randint(50, 100, size)
    digital = np.random.randint(40, 100, size)

    growth = (
        0.2 * energy +
        0.2 * waste +
        0.2 * csr +
        0.2 * employee +
        0.2 * digital
    ) / 5 + np.random.normal(0, 2, size)

    df = pd.DataFrame({
        "Energy": energy,
        "Waste": waste,
        "CSR": csr,
        "Employee": employee,
        "Digital": digital,
        "Growth": growth
    })

    return df


def train_model(df):
    X = df.drop("Growth", axis=1)
    y = df["Growth"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    return model, r2, mse


def predict_growth(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]
