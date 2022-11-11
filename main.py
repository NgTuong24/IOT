import numpy as np
import pandas as pd
from data_analysis import DataAnalysis
import seaborn as sns
from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
from models import Models

if __name__ == "__main__":
    # data = pd.read_csv("./dataset/diabetes.csv", sep="\t")
    data = pd.read_csv("./dataset/AirQuality1.csv", sep=",")
    data_analysis = DataAnalysis()
    # print(data)
    df = pd.DataFrame(data)
    # xoa gia tri ko hop le trong label
    label = df.columns[len(df.columns)-1]
    df = df[df[label] > 0]
    # print(df)

    # # Drop Feature
    # df = data_analysis.drop_feature(df=df, feature="Date")
    # df = data_analysis.drop_feature(df=df, feature="Time")

    # # Data format
    df = data_analysis.replace_val(df=df, val_old=',', val_new='.')
    for index, attribute in enumerate(df.columns, 0):
        # print(df.dtypes[index])
        # print(attribute)
        if df.dtypes[index] == "object":
            df = data_analysis.convert_object_to_float(df=df, attribute=attribute)

    # #
    df = data_analysis.replace_val(df=df, val_old=-200, val_new=np.nan)
    # data_analysis.show_percent_missing_values(df=df)
    # drop attribute has missing data > 80%
    state, list_attribute_has_nan_value = data_analysis.check_null_data(df=df)
    if state:
        list_drop = []
        for attribute in list_attribute_has_nan_value:
            if data_analysis.calculate_percent_missing_data(df=df, attribute=attribute[0]) >= 80:
                df = data_analysis.drop_feature(df=df, feature=attribute[0])
                list_drop.append(attribute)
        list_attribute_has_nan_value = list(set(list_attribute_has_nan_value) ^ set(list_drop))

    # # Removing Outliers
    # print(data_analysis.limit_value_feature(df=df))
    if state:
        df = data_analysis.filling_missing_values(df=df, list_feature_has_nan_value=list_attribute_has_nan_value)
    df = data_analysis.data_cleaning(df=df)
    # reset index
    df = data_analysis.reset_index(df=df)
    # print(df)

    # # Plotting correlation matrix
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    list_correlation = []
    list_feature = list(df.columns)[:-1]
    value_label = df[label].values.tolist()
    for feature in list_feature:
        val_feature = df[feature].values.tolist()
        corr = data_analysis.correlation(feature=val_feature, label=value_label)
        list_correlation.append((feature, corr))
    # sort
    list_correlation = sorted(list_correlation, key=lambda x: abs(x[1]))
    print("List Correlation (Increase): ", list_correlation)

    # # models-training_testing
    model = Models(df=df)
    # model.linear()
    list_result_mes = []
    list_result_r2 = []
    list_accuracy_svm = []
    for index in range(len(list_correlation) - 1):
        value = list_correlation[index]
        print(f"lần xóa thứ {index + 1} xóa thêm cột {value}")
        df.drop(value[0], axis=1, inplace=True)
        # print(df)
        # list_accuracy_svm.append(cov.svm())
        mes, r2 = model.linear()
        list_result_mes.append(mes)
        list_result_r2.append(r2)
    result_data_frame_linear = pd.DataFrame({"mes": list_result_mes, "r2": list_result_r2})
    print(result_data_frame_linear)




