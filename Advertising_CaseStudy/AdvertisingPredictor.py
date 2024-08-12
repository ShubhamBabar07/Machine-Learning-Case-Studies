# Advertising Predictor by Linear Regression

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def AdvertisementPredictor(data_path):

    data = pd.read_csv(data_path, index_col=0)

    print("Size of Actual dataset :", len(data))

    feature_names = ['TV', 'radio', 'newspaper']

    print("\nNames of Features :", feature_names)

    X = data[feature_names]
    y = data.sales

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/2)

    print("\nSize of Training dataset :", len(X_train))
    print("\nSize of Testing dataset :", len(X_test))

    linreg = LinearRegression()

    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    print("\nTesting set :")
    print(X_test)

    print("\nResult of Testing : ")
    print(y_pred)

    print("\nmean_squared_error : ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def main():

    AdvertisementPredictor("Advertising.csv")


if __name__ == "__main__":
    main()
