import pandas as pd
from copy import copy
from statistics import mean, stdev
from math import exp, sqrt, pi
from src.solver import Solver


class NaiveBayesSolver(Solver):
    def __init__(self):
        self.data_info = {}

    def __calculate_data_info(self, data: pd.DataFrame):
        for label_value in self.all_labels:
            col_info = {}
            data_by_label = data.loc[data[self.label] == label_value]
            data_by_label = data_by_label.drop(self.label, axis=1)
            for col in data_by_label.columns:
                if col in self.numerical_attrs:
                    col_info[col] = (
                        mean(data_by_label[col]),
                        stdev(data_by_label[col]),
                    )
                else:
                    unique_col_vals = data_by_label[col].unique()
                    unique_vals_probs = {}
                    for val in unique_col_vals:
                        unique_vals_probs[val] = len(
                            data_by_label.loc[data_by_label[col] == val]
                        ) / len(data_by_label[col])
                    col_info[col] = unique_vals_probs
            self.data_info[label_value] = (col_info, len(data_by_label[col]))

    def __calculate_value_probability(
        self,
        value,
        column,
        info,
    ):
        if column not in self.numerical_attrs:
            return info[value]
        else:
            val_mean = info[0]
            val_stdev = info[1]
            exponent = exp(-((value - val_mean) ** 2 / (2 * val_stdev**2)))
            return (1 / (sqrt(2 * pi) * val_stdev)) * exponent

    def __calculate_probabilities(self, record):
        total_rows = sum(self.data_info[label][1] for label in self.data_info)
        probs = {}
        for label, label_info in self.data_info.items():
            probs[label] = label_info[1] / total_rows
            for i in range(len(label_info)):
                value = record[i]
                col_name = list(label_info[0].keys())[i]
                info = label_info[0][col_name]
                probs[label] *= self.__calculate_value_probability(
                    value, col_name, info
                )

        return probs

    def __predict_row(self, row):
        probs = self.__calculate_probabilities(row)
        best_label, best_prob = None, -1
        for label, probability in probs.items():
            if probability > best_prob:
                best_label = label
                best_prob = probability
        return best_label

    def __calculate_accuracy(self, y, predictions):
        return (predictions == y).sum() / len(y)

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            predictions.append(self.__predict_row(row))
        return predictions

    def fit(self, X, y, numerical_attrs, label):
        train_data = pd.concat([X, y], axis=1)
        self.numerical_attrs = numerical_attrs
        self.label = label
        self.all_labels = train_data[label].unique()
        self.__calculate_data_info(train_data)

    def evaluate(self, train_X, train_y, val_X, val_y, numerical_attrs, label):
        self.data_info = {}
        self.fit(train_X, train_y, numerical_attrs, label)
        predictions = self.predict(val_X)
        return self.__calculate_accuracy(val_y, predictions)

    def evaluate_cross_validation(self, X_split, y_split, numerical_attrs, label):
        accs = []
        for fold in range(len(X_split)):
            curr_split_X = copy(X_split)
            curr_split_y = copy(y_split)
            val_X = curr_split_X.pop(fold)
            val_y = curr_split_y.pop(fold)
            train_X = pd.concat(curr_split_X, axis=0)
            train_y = pd.concat(curr_split_y, axis=0)
            accs.append(
                self.evaluate(train_X, train_y, val_X, val_y, numerical_attrs, label)
            )
        return accs
