from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')


class DataAnalysis:
    def __init__(self):
        pass

    def reset_index(self, df: DataFrame):
        df = df.reset_index(drop=True)
        return df

    def drop_feature(self, df: DataFrame, feature: str):
        df.drop(feature, axis=1, inplace=True)
        print(f"Drop {feature} successfully.")
        return df

    def replace_val(self, df: DataFrame, val_old, val_new):
        df.replace(to_replace=val_old, value=val_new, regex=True, inplace=True)
        return df

    def convert_object_to_float(self, df: DataFrame, attribute: str):
        df[attribute] = pd.to_numeric(df[attribute], errors='coerce')
        return df

    def check_null_data(self, df: DataFrame):
        list_feature_has_value_is_null = []
        check_null = df.isna().sum()
        # print(check_null)
        for index, value in enumerate(check_null, 0):
            if value > 0:
                list_feature_has_value_is_null.append((df.columns[index], value))
        if len(list_feature_has_value_is_null) != 0:
            list_feature_has_value_is_null = sorted(list_feature_has_value_is_null,
                                                    key=lambda x: x[1], reverse=True)
            return True, list_feature_has_value_is_null
        return False, None

    def show_percent_missing_values(self, df: DataFrame):
        plt.figure(figsize=(12, 6))
        missing_values = round(df.isna().sum() * 100 / len(df), 2)
        # print(missing_values)
        # lấy các giá trị > 0
        missing_values = missing_values[missing_values > 0]
        missing_values.sort_values(inplace=True, ascending=False)
        sns.barplot(x=missing_values.index, y=missing_values.values)
        plt.title('Missing Values %')
        plt.show()

    def calculate_percent_missing_data(self, df: DataFrame, attribute: str):
        percent = df[attribute].isna().sum() / len(df[attribute])
        # print(f'The {attribute} sensor has {percent * 100}% of missing data.')
        return percent * 100

    def limit_value_feature(self, df: DataFrame):
        # 3-Standard Deviation method
        list_value_limit = []
        attribute = list(df.columns)
        features = attribute[:-1]
        for feature in features:
            upper = df[f'{feature}'].mean() + 3 * df[f'{feature}'].std()
            lower = df[f'{feature}'].mean() - 3 * df[f'{feature}'].std()
            list_value_limit.append((feature, lower, upper))
        return list_value_limit

    def filling_missing_values(self, df: DataFrame, list_feature_has_nan_value: list):
        list_value_limit = self.limit_value_feature(df)
        for feature in list_feature_has_nan_value:
            # print(feature)
            # feature : ("feature_name", count_miss_value)
            feature_name = feature[0]
            lower = None
            upper = None
            for limit_value in list_value_limit:
                if limit_value[0] == feature_name:
                    lower = limit_value[1]
                    upper = limit_value[2]
                    break
            feature_mean = df[(df[feature_name] >= lower) & (df[feature_name] <= upper)][feature_name].mean()
            df[feature_name].fillna(feature_mean, inplace=True)
        # feature_mean = df[(df['C6H6(GT)'] >= 0) & (df['C6H6(GT)'] <= 1000)]["C6H6(GT)"].mean()
        # print(feature_mean)
        # df["C6H6(GT)"].fillna(feature_mean, inplace=True)
        return df

    def data_cleaning(self, df: DataFrame):
        list_limit_value = self.limit_value_feature(df)
        for value in list_limit_value:    # (feature, lower, upper)
            feature, lower, upper = value
            df = df[df[feature] >= lower]
            df = df[df[feature] <= upper]
        return df

    def covariance(self, feature: list, label: list):
        mean_feature = sum(feature) / float(len(feature))
        mean_label = sum(label) / float(len(label))
        sub_feature = [float(i) - mean_feature for i in feature]
        sub_label = [float(i) - mean_label for i in label]
        numerator = sum([sub_feature[i] * sub_label[i] for i in range(len(sub_feature))])
        denominator = len(feature) - 1
        cov = numerator / denominator
        return cov

    def correlation(self, feature: list, label: list):
        mean_feature = sum(feature) / float(len(feature))
        mean_label = sum(label) / float(len(label))
        sub_feature = [float(i) - mean_feature for i in feature]
        sub_label = [float(i) - mean_label for i in label]
        numerator = sum([sub_feature[i] * sub_label[i] for i in range(len(sub_feature))])
        std_deviation_feature = sum([sub_feature[i] ** 2.0 for i in range(len(sub_feature))])
        std_deviation_y = sum([sub_label[i] ** 2.0 for i in range(len(sub_label))])
        denominator = (std_deviation_feature * std_deviation_y) ** 0.5
        cor = numerator / denominator
        return cor


