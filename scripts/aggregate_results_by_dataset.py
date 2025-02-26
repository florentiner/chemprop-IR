from argparse import ArgumentParser
import os

EXPERIMENTS = [
    'random_forest',
    'ffn_morgan',
    'ffn_morgan_count',
    'ffn_rdkit',
    'mayr_et_al',
    'default',
    'features_no_opt',
    'hyperopt_eval',
    'hyperopt_ensemble',
    'undirected',
    'atom_messages'
]
ORDER = {f'417_{name}': index for index, name in enumerate(EXPERIMENTS)}


def aggregate_results_by_dataset(dataset: str, ckpt_dir: str, split_type: str):
    results = {experiment: [] for experiment in EXPERIMENTS}

    # Collect results
    for experiment in EXPERIMENTS:
        exp_dir = os.path.join(ckpt_dir, f'417_{experiment}', dataset, split_type)

        if not os.path.exists(exp_dir):
            continue

        # Collect verbose.log files
        paths = []
        for root, _, files in os.walk(exp_dir):
            paths += [os.path.join(root, fname) for fname in files if fname == 'verbose.log']

        for path in paths:
            with open(path) as rf:
                # Get last line
                for line in rf:
                    last_line = line

                # e.g. Overall test rmse = 0.939207 +/- 0.000000
                try:
                    last_line = last_line.strip().split('=')[1]
                    last_line = last_line.split('+')[0]
                    results[experiment].append(float(last_line.strip()))
                except (IndexError, ValueError):
                    print(f'Invalid path "{path}"')

    # Print results

    max_num_folds = max(len(res) for res in results.values())



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to collect results for')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Path to directory containing all checkpoints '
                             'which will be walked to find checkpoints for the dataset')
    parser.add_argument('--split_type', type=str, required=True, choices=['random', 'scaffold'],
                        help='"random" or "scaffold"')
    args = parser.parse_args()

    aggregate_results_by_dataset(
        dataset=args.dataset,
        ckpt_dir=args.ckpt_dir,
        split_type=args.split_type
    )
