import os
import csv

import numpy as np
import random


def get_all_data(test_split=0.1, interval=3, verbose=False):
    '''
    Given: test_split: float, interval: int
    Return: x_train, y_train, x_test, y_test, labels
    - each data point contains continuous tensile data over an interval number of increments
    and there is no overlap between a the increments in a training point and those in any test points
    - labels contains the mapping between y outputs and the name of the material predicted
    '''
    filenames = os.listdir("data/")
    labels = {}
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    random.seed(11)
    for i in range(len(filenames)):
        filename = filenames[i]
        label = filename[:-4]
        labels[i] = label
        with open(os.path.join("data", filename), 'r') as fp:
            csvreader = csv.reader(fp)
            columns = next(csvreader)
            rows = []
            for row in csvreader:
                row = row[0].strip().split(';')
                rows.append(row)

        # 0 <= x < 1
        x = random.random()

        # test set bounds: [test_start, test_end)
        min_test_start = int(interval)
        max_test_start = int(len(rows) * (1 - test_split))
        test_start = int(x * (max_test_start - min_test_start)) + min_test_start
        test_end = test_start + int((len(rows) * test_split))
        for j in range(interval - 1, test_start):
            dtpt = np.array(rows[j - interval + 1: j + 1], dtype=np.float64).flatten()
            x_train.append(dtpt)
            y_train.append(i)

        for j in range(test_start + interval - 1, test_end):
            dtpt = np.array(rows[j - interval + 1: j + 1], dtype=np.float64).flatten()
            x_test.append(dtpt)
            y_test.append(i)

        if test_end + interval - 1 < len(rows):
            for j in range(test_end + interval - 1, len(rows)):
                dtpt = np.array(rows[j - interval + 1: j + 1], dtype=np.float64).flatten()
                x_train.append(dtpt)
                y_train.append(i)

        if verbose:
            print(f'Class {i}:')
            print(f'Number of train examples: {len(x_train)}')
            print(f'Number of test examples: {len(x_test)}')
            print(f'Test Start: {test_start}, Test End: {test_end}\n')

    if verbose:
        print(labels)

    return (np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), labels)


if __name__ == "__main__":
    get_all_data(verbose=True)
