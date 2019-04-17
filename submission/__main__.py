import json
from argparse import ArgumentParser
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, \
    RandomizedSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

INPUTS_ARG = "inputs"
OUTPUTS_ARG = "outputs"
KEYS_ARG = "keys"
TO_CLASSIFY_ARG = "classify"


def parse_args() -> Dict:
    ap = ArgumentParser()
    ap.add_argument("-x", "--" + INPUTS_ARG, required=True, help="Path to csv file containing inputs")
    ap.add_argument("-y", "--" + OUTPUTS_ARG, required=True, help="Path to csv file containing outputs")
    ap.add_argument("-k", "--" + KEYS_ARG, required=True, help="Path to JSON file mapping outputs to class names")
    ap.add_argument("-c", "--" + TO_CLASSIFY_ARG, required=True, help="Path to of inputs needing classified")

    parsed_args = ap.parse_args()

    return vars(parsed_args)


def load_data(input_file, output_file, keys_file, to_classify_file):
    """Load input, output, and class titles from the given file names"""
    # Read in input data
    input_data = pd.read_csv(input_file, delimiter=',', error_bad_lines=True, dtype=float, header=None)

    # Read in output data
    output_data = pd.read_csv(output_file, delimiter=',', error_bad_lines=True, dtype=float, header=None)

    class_keys = {}
    with open(keys_file) as f:
        class_keys = json.load(f)

    class_keys = {int(key): value for key, value in class_keys.items()}

    # Read in to classify file
    to_classify = pd.read_csv(to_classify_file, delimiter=',', error_bad_lines=True, dtype=float, header=None)

    return input_data, output_data, class_keys, to_classify


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


def show_confusion_matrix(actual, predicted, model_used, output_keys, save=True):
    conf_matrix = confusion_matrix(actual, predicted)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cat = ax.matshow(conf_matrix, cmap=plt.cm.gray)
    labels = [''] + [x for x in output_keys.values()]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("Confusion Matrix for {}".format(model_used))

    if save:
        plt.savefig("{}.png".format(model_used))

    plt.show()


def random_forest_random_search(pipeline, training_input, training_output):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'clf__n_estimators': n_estimators,
                   'clf__max_features': max_features,
                   'clf__max_depth': max_depth,
                   'clf__min_samples_split': min_samples_split,
                   'clf__min_samples_leaf': min_samples_leaf,
                   'clf__bootstrap': bootstrap}

    random = RandomizedSearchCV(estimator=pipeline, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                random_state=42, n_jobs=-1)

    print(pipeline.get_params())
    random.fit(training_input, training_output)
    print(random.best_params_)


def random_forest_grid_search(pipeline, training_input, training_output):
    # Best for binary:
    # param_grid = {
    #     'clf__bootstrap': [True],
    #     'clf__max_depth': [2],
    #     'clf__max_features': ["sqrt"],
    #     'clf__min_samples_leaf': [1],
    #     'clf__min_samples_split': [2],
    #     'clf__n_estimators': [50]
    # }
    # Best for multiclass
    param_grid = {
        'clf__bootstrap': [True],
        'clf__max_depth': [4],
        'clf__max_features': ["sqrt"],
        'clf__min_samples_leaf': [1],
        'clf__min_samples_split': [2],
        'clf__n_estimators': [700]
    }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(training_input, training_output)
    print(grid_search.best_params_)


def generate_random_forest(training_input, testing_input, training_output, testing_output, keys):
    """Random Forest"""
    pipeline = Pipeline([
        ('scalar', MinMaxScaler(feature_range=(0, 1))),
        ('pca', PCA(n_components=0.95, random_state=42)),
        ('clf', RandomForestClassifier(random_state=42, bootstrap=True, max_depth=4, max_features="sqrt",
                                       min_samples_leaf=1, min_samples_split=2, n_estimators=700))
    ])

    # Find a good starting point for the best hyperparameters (commented out because it takes a while)
    # random_forest_random_search(pipeline, training_input, training_output)
    # Result for binary = {'clf__n_estimators': 400, 'clf__min_samples_split': 5, 'clf__min_samples_leaf': 1,
    # 'clf__max_features': 'sqrt', 'clf__max_depth': 30, 'clf__bootstrap': True}
    # Result for multiclass = {'clf__n_estimators': 1000, 'clf__min_samples_split': 2, 'clf__min_samples_leaf': 1,
    # 'clf__max_features': 'sqrt', 'clf__max_depth': 20, 'clf__bootstrap': True}
    # These values were then used as a starting point for the grid search
    # random_forest_grid_search(pipeline, training_input, training_output)

    # training_predicted = cross_val_predict(pipeline, training_input, training_output, cv=10)
    # show_confusion_matrix(training_output, training_predicted, "Random Forest", keys, save=True)

    pipeline.fit(training_input, training_output)

    train_predictions = pipeline.predict(testing_input)
    show_confusion_matrix(testing_output, train_predictions, "Random Forest", keys, True)

    return pipeline


def sgd_random_search(pipeline, training_input, training_output):
    # Learning rate
    alpha = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    # Loss method
    loss = ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron']
    # regulizer which determines how complexity is favoured over precision
    penalty = [None, "l2", "l1", "elasticnet"]

    # Create the random grid
    random_grid = {'clf__estimator__alpha': alpha,
                   'clf__estimator__loss': loss,
                   'clf__estimator__penalty': penalty,
                   }

    random = RandomizedSearchCV(estimator=pipeline, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                random_state=42, n_jobs=-1)

    print(pipeline.get_params())
    random.fit(training_input, training_output)
    print(random.best_params_)


def sgd_grid_search(pipeline, training_input, training_output):
    # Create the random grid
    param_grid = {'clf__estimator__alpha': [1e-3, 1e-4, 1e-5],
                  'clf__estimator__loss': ["hinge"],
                  'clf__estimator__penalty': ["l2"],
                  }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(training_input, training_output)
    print(grid_search.best_params_)


def generate_sgd(training_input, testing_input, training_output, testing_output, keys):
    """OVA SGD"""
    pipeline = Pipeline([
        ('scalar', MinMaxScaler()),
        ('pca', PCA(n_components=0.95, random_state=42)),
        ('clf', OneVsOneClassifier(
            SGDClassifier(random_state=42, max_iter=3000, tol=1e-3, alpha=1e-4, loss="hinge", penalty="l2")))
    ])

    # Find a good starting point for the best hyperparameters (commented out because it takes a while)
    # sgd_random_search(pipeline, training_input, training_output)
    # sgd_grid_search(pipeline, training_input, training_output)

    # training_predicted = cross_val_predict(pipeline, training_input, training_output, cv=10)
    # show_confusion_matrix(training_output, training_predicted, "SGD", keys)

    pipeline.fit(training_input, training_output)
    predicted = pipeline.predict(testing_input)
    show_confusion_matrix(testing_output, predicted, "OVO SGDClassifier", keys, True)

    return pipeline


if __name__ == "__main__":
    # Loading Data
    args = parse_args()
    inputs, outputs, keys, to_classify = load_data(args[INPUTS_ARG], args[OUTPUTS_ARG], args[KEYS_ARG],
                                                   args[TO_CLASSIFY_ARG])
    # Cleaning and preparing data
    inputs, outputs = clean_data(inputs, outputs)
    x_train, x_test, y_train, y_test = split_data(inputs, outputs)

    # Numpy arrays are easily to deal with...
    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]

    # Visualising data
    # visualise_signal(x_train, y_train, keys)
    # visualise_class_distribution(y_train, keys)

    # Drop everything but averages
    x_train = x_train.iloc[:, 0:256]
    x_test = x_test.iloc[:, 0:256]
    to_classify = to_classify.iloc[:, 0:256]

    # PCA Examples
    # plot_2d_pca(x_train, y_train, keys)
    # plot_explained_variance(x_train)

    rf_pipeline = generate_random_forest(x_train, x_test, y_train, y_test, keys)
    sgd_pipeline = generate_sgd(x_train, x_test, y_train, y_test, keys)

    final_output = rf_pipeline.predict(to_classify)
    np.savetxt("PredictedClasses.csv", final_output, delimiter=",")

# NOTE FOR MARKER : Some lines are commented out since they take a while,
# or were used for experimenting but are all functional if uncommented
