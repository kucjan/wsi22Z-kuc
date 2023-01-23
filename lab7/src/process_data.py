import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def process_data(filename):
    dataset = pd.read_csv(filename, delimiter=";")
    dataset = dataset.drop("id", axis=1)
    print(dataset.hist())

    dataset, missing_values_columns = missing_values(dataset, True, True)

    print(missing_values_columns)

    return dataset


def missing_values(dataset, space_to_nan, drop_empty):
    if space_to_nan:
        dataset = dataset.replace(r"^\s*$", np.NaN, regex=True)

    if drop_empty:
        dataset = dataset.drop(dataset.index[dataset.isnull().all(axis=1)]).reset_index(
            drop=True
        )

    missing_values = dataset.isnull().sum()

    missing_values_percent = 100 * dataset.isnull().sum() / len(dataset)

    missing_values_df = pd.concat([missing_values, missing_values_percent], axis=1)

    missing_values_columns = missing_values_df.rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )

    return dataset, missing_values_columns


def dataset_split(dataset, label):
    # dataset split: 60% x 20% x 20%
    train_data, valid_data, test_data = np.split(
        dataset.sample(frac=1), [int(0.6 * len(dataset)), int(0.8 * len(dataset))]
    )

    train_data_X = train_data.drop(label, axis=1)
    train_data_y = train_data[label]
    valid_data_X = valid_data.drop(label, axis=1)
    valid_data_y = valid_data[label]
    test_data_X = test_data.drop(label, axis=1)
    test_data_y = test_data[label]

    return (
        train_data_X,
        train_data_y,
        valid_data_X,
        valid_data_y,
        test_data_X,
        test_data_y,
    )


def cross_validation_split(dataset, label, n_folds):
    X = dataset.drop(label, axis=1)
    y = dataset[label]

    cross_data_X, test_data_X, cross_data_y, test_data_y = train_test_split(
        X, y, test_size=0.2
    )

    dataset_split_X = []
    dataset_split_y = []
    fold_size = int(len(cross_data_X) / n_folds)

    for i in range(0, len(cross_data_X), fold_size):
        dataset_split_X.append(cross_data_X[i : i + fold_size])
        dataset_split_y.append(cross_data_y[i : i + fold_size])

    return dataset_split_X, dataset_split_y, test_data_X, test_data_y
