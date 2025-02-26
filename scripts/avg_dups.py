"""Averages the target values for duplicate smiles strings. (Only used for regression datasets.)"""

from argparse import ArgumentParser
from collections import defaultdict
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_data, get_header


def average_duplicates(args):
    """Averages duplicate data points in a dataset."""
    'Loading data')
    header = get_header(args.data_path)
    data = get_data(path=args.data_path)

    # Map SMILES string to lists of targets
    smiles_in_order = []
    smiles_to_targets = defaultdict(list)
    for smiles, targets in zip(data.smiles(), data.targets()):
        smiles_to_targets[smiles].append(targets)
        if len(smiles_to_targets[smiles]) == 1:
            smiles_in_order.append(smiles)

    # Find duplicates
    duplicate_count = 0
    stds = []
    new_data = []
    for smiles in smiles_in_order:
        all_targets = smiles_to_targets[smiles]
        duplicate_count += len(all_targets) - 1
        num_tasks = len(all_targets[0])

        targets_by_task = [[] for _ in range(num_tasks)]
        for task in range(num_tasks):
            for targets in all_targets:
                if targets[task] is not None:
                    targets_by_task[task].append(targets[task])

        stds.append([np.std(task_targets) if len(task_targets) > 0 else 0.0 for task_targets in targets_by_task])
        means = [np.mean(task_targets) if len(task_targets) > 0 else None for task_targets in targets_by_task]
        new_data.append((smiles, means))

    # Save new data
    with open(args.save_path, 'w') as f:
        f.write(','.join(header) + '\n')

        for smiles, avg_targets in new_data:
            f.write(smiles + ',' + ','.join(str(value) if value is not None else '' for value in avg_targets) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--save_path', type=str,
                        help='Path where average data CSV file will be saved')
    args = parser.parse_args()

    average_duplicates(args)
