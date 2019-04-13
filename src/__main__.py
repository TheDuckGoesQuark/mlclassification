from typing import Dict, Tuple

import pandas as pd
import json
from sklearn.model_selection import train_test_split

from argparse import ArgumentParser

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
    inputs = pd.read_csv(input_file, delimiter=',', error_bad_lines=True, dtype=float, header=None)

    # Read in output data
    outputs = pd.read_csv(output_file, delimiter=',', error_bad_lines=True, dtype=float, header=None)

    with open(keys_file) as f:
        keys = json.load(f)

    return inputs, outputs, keys


def split_data(input_df, output_df):
    """Split input and outputs using stratification with a 80%-20% split"""
    return train_test_split(input_df, output_df, test_size=0.2, random_state=42,
                            stratify=output_df)


def clean_data(input_df, output_df):
    """Removes any rows with empty or missing values"""
    all_rows = pd.concat([input_df, output_df], axis=1)
    all_rows.dropna(how='all', inplace=True)
    # TODO split all rows back to input cols and output col
    input_df = all_rows[:, :all_rows.shape[1]]
    output_df = all_rows[:, all_rows.shape[1]:]
    return input_df, output_df


if __name__ == "__main__":
    args = parse_args()
    inputs, outputs, keys = load_data(args[INPUTS_ARG], args[OUTPUTS_ARG], args[KEYS_ARG])
    inputs, outputs = clean_data(inputs, outputs)
    x_train, x_test, y_train, y_test = split_data(inputs, outputs)
