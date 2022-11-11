from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import numpy as np


class Models:
    def __init__(self, df: DataFrame):
        self.df = df
        pass

    def linear(self):
        label = self.df.columns[len(self.df.columns) - 1]
        x = self.df.drop(label, axis=1)
        y = self.df[label]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        y_pr = lr.predict(x_test)
        # result = pd.DataFrame({"Dự đoán": y_pr, "Thực tế": y_test})
        mes = self.mse(actual=y_test, predicted=y_pr)
        r2 = self.r2_score(y_test=y_test, y_pre=y_pr)
        print(f"mes:{mes} r2: {r2}")
        return mes, r2

    def train_test_split(self, x, y, text_size=0.2, random_state=42):

        return

    def mse(self, actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)
        res = 0
        for i in range(len(actual)):
            res += pow(actual[i] - predicted[i], 2)
        res /= len(actual)
        return res

    def r2_score(self, y_test, y_pre):
        actual = np.array(y_test)
        predicted = np.array(y_pre)
        sum_E1 = 0
        mean_y = actual.mean()
        for i in range(len(actual)):
            sum_E1 += pow(actual[i] - predicted[i], 2)
        sum_E2 = 0
        for i in range(len(actual)):
            sum_E2 += pow(actual[i] - mean_y, 2)
        return 1 - sum_E1 / sum_E2