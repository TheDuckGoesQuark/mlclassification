from typing import Dict

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict

from argparse import ArgumentParser

from sklearn.multiclass import OneVsRestClassifier

INPUTS_ARG = "inputs"
OUTPUTS_ARG = "outputs"
KEYS_ARG = "keys"


def parse_args() -> Dict:
    ap = ArgumentParser()
    ap.add_argument("-x", "--" + INPUTS_ARG, required=True, help="Path to csv file containing inputs")
    ap.add_argument("-y", "--" + OUTPUTS_ARG, required=True, help="Path to csv file containing outputs")
    ap.add_argument("-k", "--" + KEYS_ARG, required=True, help="Path to JSON file mapping outputs to class names")

    parsed_args = ap.parse_args()

    return vars(parsed_args)


def load_data(input_file, output_file, keys_file):
    """Load input, output, and class titles from the given file names"""
    # Read in input data
    input_data = pd.read_csv(input_file, delimiter=',', error_bad_lines=True, dtype=float, header=None)

    # Read in output data
    output_data = pd.read_csv(output_file, delimiter=',', error_bad_lines=True, dtype=float, header=None)

    class_keys = {}
    with open(keys_file) as f:
        class_keys = json.load(f)

    class_keys = {int(key): value for key, value in class_keys.items()}

    return input_data, output_data, class_keys


def split_data(input_df, output_df):
    """Split input and outputs using stratification with a 80%-20% split"""
    return train_test_split(input_df, output_df, test_size=0.2, random_state=42,
                            stratify=output_df)


def clean_data(input_df, output_df):
    """Removes any rows with empty or missing values"""
    all_rows = pd.concat([input_df, output_df], axis=1)
    all_rows.dropna(how='all', inplace=True)
    input_df = all_rows.iloc[:, 0:input_df.shape[1]]
    output_df = all_rows.iloc[:, input_df.shape[1]:all_rows.shape[1]]
    return input_df, output_df


def plot_class_scatter(rows_of_class, class_name, max_value, min_value):
    """Plot scatter graphs for the functions applied to the first channel"""
    fig = plt.figure(figsize=(30, 5))
    fig.suptitle("Components for class {}".format(class_name))
    function_to_channel_plots = {function_name: [] for function_name in ["Mean", "Min", "Max"]}
    n_plots = 1
    # For each function
    for function_idx, function_name in enumerate(function_to_channel_plots):
        # For each channel
        for channel_idx in range(0, 4):
            plot = fig.add_subplot(1, 14, n_plots + function_idx)
            channel_number = ((n_plots - 1) % 4) + 1
            plot.set_title("{} of Channel {}".format(function_name, channel_number))
            plot.set_xlabel("Components")
            # Only need title for first graph for each function
            if channel_idx == 0:
                plot.set_ylabel("{} of 100 pulses".format(function_name))

            plot.set_ylim((min_value, max_value))
            function_to_channel_plots[function_name].append(plot)
            n_plots += 1

    components_per_function = 256
    components_per_channel = 64
    for index, row in rows_of_class.iterrows():
        for function_idx, (function, channel_plots) in enumerate(function_to_channel_plots.items()):
            for channel_idx, channel_plot in enumerate(channel_plots):
                x = np.arange(0, components_per_channel)
                start = (function_idx * components_per_function) + (channel_idx * components_per_channel)
                end = start + components_per_channel
                y = row[start:end]
                channel_plot.scatter(x, y, alpha=0.8)

    plt.savefig("{}.png".format(class_name))


def visualise_signal(input_rows_for_class, output_rows_for_class, class_keys):
    all_rows = pd.merge(input_rows_for_class, output_rows_for_class, left_index=True, right_index=True)
    classes = all_rows.iloc[:, -1].unique()
    max_value = input_rows_for_class.values.max()
    min_value = input_rows_for_class.values.min()
    for class_enum in classes:
        # Get indices of all rows for that class
        indices_of_class = all_rows.index[all_rows.iloc[:, -1] == class_enum]
        # Get rows using those indices
        class_rows = all_rows.loc[indices_of_class]
        # Plot components for each class
        plot_class_scatter(class_rows, class_keys[class_enum], max_value, min_value)


def visualise_class_distribution(y_vals, y_keys):
    fig, ax = plt.subplots()
    fig.suptitle("Frequency of Each Class in Training Data Set")
    ax.set_xlabel("Class")
    ax.set_ylabel("Frequency in Data Set")
    counts = y_vals.iloc[:, 0].value_counts().to_dict()
    counts = {y_keys[key]: value for key, value in counts.items()}
    plt.bar(counts.keys(), counts.values())
    plt.savefig("frequencies.png")


def normalise_data(x_vals):
    min_val = x_vals.values.min()
    max_val = x_vals.values.max()
    return (x_vals - min_val) / (max_val - min_val)


def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive rate")
    plt.ylabel("True Positive Rate")


def apply_model_one(training_input, testing_input, training_output, testing_output, keys):
    """Random Forest"""
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=20)
    rf_classifier.fit(training_input, training_output.values[:, 0])
    predicted = rf_classifier.predict(testing_input)
    evaluate(predicted, testing_output.values[:, 0])


def apply_model_two(training_input, testing_input, training_output, testing_output, keys):
    sgd_classifier = OneVsRestClassifier(SGDClassifier(random_state=42))
    sgd_classifier.fit(training_input, training_output.values[:, 0])
    predicted = sgd_classifier.predict(testing_input)
    evaluate(predicted, testing_output.values[:, 0])


def evaluate(predicted, actual):
    print("Accuracy: {}".format(accuracy_score(actual, predicted)))
    print(classification_report(actual, predicted))
    conf_matrix = confusion_matrix(actual, predicted)
    print(conf_matrix)
    plt.matshow(conf_matrix, cmap=plt.cm.gray)


def plot_2d_pca(training_input, training_output, output_keys):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(training_input)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'])
    all_rows = pd.merge(principal_df, training_output, left_index=True, right_index=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('2 Component PCA')

    colours = iter(plt.cm.rainbow(np.linspace(0, 0.9, len(output_keys))))
    for target in output_keys:
        # Get indices of all rows for that class
        indices_of_class = all_rows.index[all_rows.iloc[:, -1] == target]
        ax.scatter(all_rows.loc[indices_of_class, 'PC 1']
                   , all_rows.loc[indices_of_class, 'PC 2']
                   , c=next(colours)
                   , s=50)

    ax.legend(output_keys.values())
    ax.grid()
    plt.savefig("2dpca.png")
    print(pca.explained_variance_ratio_)


def plot_explained_variance(training_input):
    example_pca = PCA(n_components=0.9999999)
    example_pca.fit_transform(training_input)

    cumsum = np.cumsum(example_pca.explained_variance_ratio_)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Explained Variance')
    ax.set_title('Explained Variance with Increasing Dimensions')
    ax.plot(cumsum)
    threshold = 0.95
    ax.axhline(threshold, linestyle="--", color="black")

    plt.savefig("explainedvariance.png")


if __name__ == "__main__":
    # Loading Data
    args = parse_args()
    inputs, outputs, keys = load_data(args[INPUTS_ARG], args[OUTPUTS_ARG], args[KEYS_ARG])
    # Cleaning and preparing data
    inputs, outputs = clean_data(inputs, outputs)
    x_train, x_test, y_train, y_test = split_data(inputs, outputs)

    # Visualising data
    # visualise_signal(x_train, y_train, keys)
    # visualise_class_distribution(y_train, keys)
    # Drop everything but averages
    x_train = x_train.iloc[:, 0:256]
    x_test = x_train.iloc[:, 0:256]
    x_train = normalise_data(x_train)

    # PCA Examples
    # plot_2d_pca(x_train, y_train, keys)
    # plot_explained_variance(x_train)
    # PCA Used
    pca = PCA(n_components=0.95, random_state=42)
    x_train_reduced = pca.fit_transform(x_train)
    x_test_reduced = pca.transform(x_test)

    # Apply models
    apply_model_one(x_train_reduced, x_test_reduced, y_train, y_test, keys)
    apply_model_two(x_train_reduced, x_test_reduced, y_train, y_test, keys)
